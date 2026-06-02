from algos.strassen_winograd_level1 import strassen_winograd
from dsl import *
import argparse
import copy
import itertools
import os
import shutil
from functools import reduce

MAX_LEN_PARTITION = 2
MAX_PRESUM_BUFFERS = 2
MAX_FUSED = 4

def get_all_set_partitions(s):
  if not s:
      yield []
      return

  first = s[0]
  remaining = s[1:]

  for p in get_all_set_partitions(remaining):
      # Option 1: Add 'first' to an existing part
      for i, part in enumerate(p):
          yield p[:i] + [part + [first]] + p[i+1:]

      # Option 2: Create a new part with 'first'
      yield [[first]] + p

class Partitioning:
  def __init__(self, common_exprs, misetnodes,
               expr_def_by_miset, presums_todo,
               a_gmem_loads, b_gmem_loads, c_gmem_loads):
    self.misetnodes = misetnodes
    self.expr_def_by_miset = expr_def_by_miset
    self.presums_todo = presums_todo #A tuple of presums to do with their use
    self.a_gmem_loads = a_gmem_loads
    self.b_gmem_loads = b_gmem_loads
    self.c_gmem_loads = c_gmem_loads
    self.fused_mi_sets = None
    self.common_exprs = common_exprs

  def init_misetnodes_wo_presum(self, mis_wo_presum):
    self.nodes_wo_presum = []
    for misetnode in self.misetnodes:
      if len(misetnode.miset) == 1 and misetnode.miset[0] in mis_wo_presum:
        self.nodes_wo_presum += [misetnode]
    return self.nodes_wo_presum

  def misetnodes_wo_presum(self):
    return self.nodes_wo_presum

  def total_gmem_loads(self):
    return self.a_gmem_loads + self.b_gmem_loads + self.c_gmem_loads

  def partitioning(self):
    return [[mi for mi in node.miset] for node in self.misetnodes]

  def set_fused_mi_sets(self, f):
    self.fused_mi_sets = f

  def get_fused_miset_for_miset(self, m):
    for fmi in self.fused_mi_sets:
      if m in fmi.misets:
        return fmi
    return m

  def __repr__(self):
    return repr(self.partitioning()) + " " + repr(self.total_gmem_loads()) + " " + \
           repr([(p[0]) for p in self.presums_todo["a"]]) + " " + \
           repr([(p[0]) for p in self.presums_todo["b"]])

  def __str__(self):
    return repr(self)
  
  def print_miset_in_order(self, order):
    for miset in order:
      print(miset.full_str())

def all_miset_of_powerto(partitioning, all_mis_len):
  for miset in partitioning:
    if len(miset) > MAX_LEN_PARTITION:
      return False
    if len(miset) == 1 or len(miset) == 2 or len(miset) == 4 or len(miset) == all_mis_len:
      pass
    else:
      return False
  return True

def atleast_one_miset_wo_presum_is_not_fused(partitioning, Ms):
  for miset in partitioning:
    if len(miset) == 1 and miset[0] in Ms:
      return True
  return False

def find_times_each_expr_is_used(C, Ms):
  times_used_dict = {}
  Cs = C.parts

  def times_used_visitor(node, *args):
    if isinstance(node, BinOp):
        if node not in Cs and node not in Ms:
            if node not in times_used_dict:
                times_used_dict[node] = 0
            times_used_dict[node]+=1
        if node.op == "@":
            return False
    return True

  C.visit(times_used_visitor)
  common_exprs = sorted(list(times_used_dict.items()),
                        key=lambda x: x[1], reverse=True)
  common_exprs = [expr[0] for expr in common_exprs]
  return common_exprs + Cs

def misetnodes_has_cycle(misetnodes):
  visited = set()
  stack = []

  def isCyclicUtil(miset):
    if miset in stack:
      return True
    
    if miset in visited:
      return False
    
    visited.add(miset)
    stack.append(miset)

    for use in miset.uses_stores:
      if isCyclicUtil(use[1]):
        return True
    
    stack.remove(miset)
    return False

  for miset in misetnodes:
    if not miset in visited and isCyclicUtil(miset):
      return True
  return False

def set_def_of_postsum_exprs(Cs, partitioning, misetnodes, common_exprs):
  #Order of set of Mis is based on the common expressions across Cis.
  #An expression that is most common will be scheduled first and a common sub-expression's output will be written
  #then the next set of Mis will read the common sub-expression and update to write the common expression,
  #then the next common and so on.
  expr_def_by_miset = {k: None for k in common_exprs}

  #We need to first order Mi where no presum is needed for both A and B
  common_exprs_with_no_presum = []
  #Find all common_exprs where all Mi do not perform any presum for both A and B
  #and put these common_exprs in the front, so that these are scheduled first
  for i,common_expr in enumerate(common_exprs):
      require_presum = False
      for mi in common_expr.get_matmul_matrices():
          if len(mi.get_base_matrices_op1()) == 1 and len(mi.get_base_matrices_op2()) == 1:
              pass
          else:
              require_presum = True
      if not require_presum:
          common_exprs_with_no_presum += [common_expr]

  for e in common_exprs_with_no_presum:
      common_exprs.remove(e)
  common_exprs = common_exprs_with_no_presum + common_exprs
  #Assign def-by matmul to leafs of the postsum expression tree 
  for i,common_expr in enumerate(common_exprs):
    # if an expr's op is output of Matmul (m0,m1,m2,..etc) set the definition of this op  
    if isinstance(common_expr.op1, Matrix):
      for partIdx, part in enumerate(partitioning):
        if common_expr.op1 in part:
            expr_def_by_miset[common_expr.op1] = misetnodes[partIdx]
    if isinstance(common_expr.op2, Matrix):
      for partIdx, part in enumerate(partitioning):
        if common_expr.op2 in part:
            expr_def_by_miset[common_expr.op2] = misetnodes[partIdx]
  
  #Assign def-by matmul to all nodes of postsum expression tree
  for i,common_expr in enumerate(common_exprs):
    both_op_require_no_presum = common_expr in common_exprs_with_no_presum
    if isinstance(common_expr.op1, Matrix) and isinstance(common_expr.op2, Matrix):
        #If both operands of an expr are matmul (i.e. no presum) outputs and both matmuls are same,
        #then this matmul computes the expr
        if expr_def_by_miset[common_expr.op1] == expr_def_by_miset[common_expr.op2]:
            expr_def_by_miset[common_expr] = expr_def_by_miset[common_expr.op1]
            expr_def_by_miset[common_expr].add_store(common_expr)
        else:
          #Otherwise if def-by is different
          expr_m = common_expr.op2
          prev_m = common_expr.op1
          if both_op_require_no_presum:
            #If common_expr.op1 and op2 do not require presum then prefer the miset with single mi
            if len(expr_def_by_miset[common_expr.op1].miset) == 1:
              prev_m = common_expr.op1
              expr_m = common_expr.op2
            elif len(expr_def_by_miset[common_expr.op2].miset) == 1:
              prev_m = common_expr.op2
              expr_m = common_expr.op1
          #TODO: If op1 do not require presum but op2 does then execute op1 before op2, i.e.:
          #op1.add_store(op1) and op2.add_load(op1) and op2.add_store(common_expr)
          expr_def_by_miset[prev_m].add_store(prev_m)
          expr_def_by_miset[prev_m].uses_stores += [(prev_m, expr_def_by_miset[expr_m])]

          expr_def_by_miset[expr_m].add_load(prev_m, misetnodes.index(expr_def_by_miset[prev_m]))
          expr_def_by_miset[expr_m].add_store(common_expr)
          expr_def_by_miset[common_expr] = expr_def_by_miset[expr_m]

    elif isinstance(common_expr.op1, BinOp) and isinstance(common_expr.op2, Matrix):
      def_by_op1 = expr_def_by_miset[common_expr.op1]
      def_by_op2 = expr_def_by_miset[common_expr.op2]
      if def_by_op1 == def_by_op2:
          expr_def_by_miset[common_expr] = def_by_op1
      else:
          expr_def_by_miset[common_expr] = def_by_op1
          expr_def_by_miset[common_expr].uses_stores += [(common_expr.op1, def_by_op1)]
          expr_def_by_miset[common_expr].add_load(common_expr.op1, misetnodes.index(def_by_op1))
      expr_def_by_miset[common_expr].add_store(common_expr)
  return expr_def_by_miset, common_exprs

def apply_presum(misetnodes):
  #Computing presums of Ais and Bis in the global memory would decrease the number 
  #of memory accesses when computing Mi. However, storing presums require more memory
  #which is not always desirable.
  #The algorithm works as follows:
  # 1. For all presum fill the presum.misets list with miset that uses these presum
  # 2. Calculate how many miset uses this presum, i.e., how common is this presum expression.
  #   2.1. For the most common presum, find if computing the presum by another miset would lead to decrease in memory for current miset.
  #   2.2. For this presum, set the presum's store by miset that comes first (what do you mean by first?)
  #   2.3. If number of presum buffers are equal to MAX_PRESUM_BUFFER, then exit.
  #   2.4. Otherwise move to the next common presum expr.
  presum_use_by_miset = {"a":{}, "b": {}}

  for i,miset in reversed(list(enumerate(misetnodes))):
    for aorb in ["a", "b"]:
      for mi in miset.miset:
        op = mi.op1 if aorb == "a" else mi.op2
        if isinstance(op, BinOp):
          for op1 in op.get_all_children():
            if isinstance(op1, BinOp):
              if op1 not in presum_use_by_miset[aorb]:
                presum_use_by_miset[aorb][op1] = []
              presum_use_by_miset[aorb][op1] += [miset]

  presum_exprs_and_uses = {"a": sorted(list(presum_use_by_miset["a"].items()), key = lambda x: len(x[1]), reverse=True),
                           "b": sorted(list(presum_use_by_miset["b"].items()), key = lambda x: len(x[1]), reverse=True)}
    # TODO: Do we need to consider if presum do not reduce mem access? probably not.
    # for presum in presum_use_by_miset:
    #   base_matrices = presum.get_base_matrices()
    #   presum_do_not_reduce_mem_access = []
    #   for miset in presum.misets:
    #       all_base_matrices = set()
    #       for mi in miset.miset:
    #           all_base_matrices.update(mi.get_base_matrices())
    #       if len(all_base_matrices - set(base_matrices)) + 1 >= len(all_base_matrices):
    #           presum_do_not_reduce_mem_access += [miset]
      # for miset in presum_do_not_reduce_mem_access:
      #     presum.miset.remove(miset)
  num_presum_buffers = 0
  presum_idx = 0
  presums_todo = {"a": [], "b": []}

  while presum_idx < len(presum_exprs_and_uses["a"]) and num_presum_buffers < MAX_PRESUM_BUFFERS:
    for aorb in ["a", "b"]:
      if num_presum_buffers < MAX_PRESUM_BUFFERS:
        presums_todo[aorb] += [presum_exprs_and_uses[aorb][presum_idx]]
        num_presum_buffers += 1
      else: break
    presum_idx += 1

  return presums_todo

def set_presum_computation(partitioning_info, mis_wo_presum):
  misetnodes_wo_presum = partitioning_info.misetnodes_wo_presum()
  for aorb in ["a", "b"]:
    for presum_todo in partitioning_info.presums_todo[aorb]:
      #Assign all presum to misets that do not do any presum for A and B operand.
      #TODO: Ideally we can distribute these presums over all misets
      miset_to_do_presum = misetnodes_wo_presum[0]
      presum = presum_todo[0]
      misets_using_presum = presum_todo[1]

      if len(misets_using_presum) > 1:
        #Add presum stores/load for the common presum to the first ordered miset
        presum_children = presum.get_all_children()
        miset_to_do_presum.presum_stores[aorb] += [presum]

        for miset in misets_using_presum:
          if miset != miset_to_do_presum:
            #When a presum is computed globally, remove all other presum loads that are not needed
            is_excess_load = [True for l in miset.presum_loads[aorb]]

            for li,load in enumerate(miset.presum_loads[aorb]):
              #Remove this load if it is used by the new presum and is not used by any other mi
              if load[0] in presum_children:
                for mi in miset.miset:
                  mi_op = (mi.op1 if aorb == "a" else mi.op2)
                  if isinstance(mi_op, BinOp):
                    if presum in mi_op.get_all_children() and load[0] in presum.get_all_children():
                      #PresumLoad is excess if presum is a child of mi_op and load[0] is a child of presum
                      pass
                    else:
                      is_excess_load[li] = False
                      break
            excess_loads = []
            for li in range(len(is_excess_load)):
              if is_excess_load[li]:
                excess_loads += [miset.presum_loads[aorb][li]]
            for l in excess_loads:
              miset.presum_loads[aorb].remove(l)

            #Add this presum load to presum's miset use
            miset.presum_loads[aorb] += [(presum, partitioning_info.misetnodes.index(miset_to_do_presum))] #order[order_idxs[0]]
      else:
        #This presum is not common and is used by only one miset
        done = False
        # if type(presum.op1) is Matrix and type(presum.op2) is Matrix:
          #When both are Matrices find miset that loads both m1 and m2 make the first miset in order 
          #do the presum.
        miset_to_do_presum.presum_stores[aorb] += [presum]
        for miset in misets_using_presum:
          miset.presum_loads[aorb] += [(presum, partitioning_info.misetnodes.index(misetnodes_wo_presum[0]))]
        # else:
        #   #Non-common presums are assigned to the miset that does the presum of immediate child
        #   #TODO: Ideally, we need to assign miset that loads/stores both the immediate presum child and other presum/Matrix.
        #   assert not (type(presum.op1) is BinOp and type(presum.op2) is BinOp)
        #   child = presum.op1 if isinstance(presum.op1, BinOp) else presum.op2
        #   miset_to_do_presum.presum_stores[aorb] += [presum]

          # for child_miset in presum_use_by_miset[aorb][child]:
            
            # if child in child_miset.presum_stores[aorb]:
            #   child_miset.presum_stores[aorb] += [presum]
            #   num_presum_buffers += 1
            #   for presum_miset in presum.misets:
            #     presum_miset.presum_loads[aorb] += [(presum, misetnodes.index(child_miset))]
            #   break
    # print(514, len(all_presum_exprs_a[start].misets), ":", order)

  return

  for aorb in ["a", "b"]:
    for miset in misetnodes:
      if miset == first_miset: continue
      for store in miset.presum_stores[aorb]:
        first_miset.presum_stores[aorb] += [store]
      for loadidx,load in enumerate(miset.presum_loads[aorb]):
        miset.presum_loads[aorb][loadidx] = (load[0], misetnodes.index(first_miset))

      #For all those Miset that does the store for presum now loads from first miset
      for mi in miset.miset:
        if type(mi.op1 if aorb == "a" else mi.op2) is BinOp:
          node = (mi.op1 if aorb == "a" else mi.op2)
          #Start from nodes at lowest level.
          #If a node at a level is available as presum then do not go to that node's children. 
          q = [(node, 0)]
          while len(q) >= 1:
            node, level = q.pop(0)
            if not isinstance(node, BinOp):
              continue
            found_node_as_presum = False    
            for store in miset.presum_stores[aorb]:
              if store == node:
                miset.add_presum_load(aorb, store, misetnodes.index(first_miset))
                found_node_as_presum = True
                break
            if not found_node_as_presum:
              q.append((node.op1, level+1))
              q.append((node.op2, level+1))

      miset.presum_stores[aorb] = []

def order_Mis(Cs, partitioning):
  fused_miset_order = []

  #Starting from the most common expr, schedule def-by miset to compute this common expr
  for i,common_expr in enumerate(partitioning.common_exprs):
    expr_order = []
    def_by_op1 = partitioning.expr_def_by_miset[common_expr.op1]
    def_by_op2 = partitioning.expr_def_by_miset[common_expr.op2]
    def_by_expr = partitioning.expr_def_by_miset[common_expr]

    if isinstance(common_expr.op1, Matrix) and def_by_op1 not in expr_order:
      expr_order += [def_by_op1]
    if isinstance(common_expr.op2, Matrix) and def_by_op2 not in expr_order:
      expr_order += [def_by_op2]
    if def_by_expr not in expr_order:
      expr_order += [def_by_expr]
    
    to_add = []
    if len(expr_order) == 1:
      to_add = expr_order
    else:
      #If def_by_op2 reads def_by_op1 then order as def_by_op1, def_by_op2.
      #Otherwise use the above expr_order
      expr_0_loads = expr_order[0].loads
      expr_1_loads = expr_order[1].loads
      is_expr_0_load_by_1 = False
      for load in expr_order[1].loads:
        if partitioning.misetnodes[load.index] == expr_order[0]:
          is_expr_0_load_by_1 = True
          break
      if is_expr_0_load_by_1:
        to_add = expr_order
      else:
        to_add = [expr_order[1], expr_order[0]]
    for m in to_add:
      fmi = partitioning.get_fused_miset_for_miset(m)
      if fmi not in fused_miset_order:
        fused_miset_order += [fmi]
  
  #Remove spurious stores
  # for i,miset in enumerate(order):
  #     stores_not_needed = []
  #     for s in miset.stores:
  #         store_not_needed = True
  #         for miset2 in order:
  #             if miset != miset2:
  #                     if len([True for l in miset2.loads if l.expr == s.expr]) > 0 and \
  #                       s.expr not in Cs:
  #                         store_not_needed = False
  #                         break
  #             if not store_not_needed: break
  #         if store_not_needed: stores_not_needed += [s]
  #     for st in stores_not_needed:
  #         miset.stores.remove(st)

  return fused_miset_order

def postsum_fusion(partitioning_info, sorted_common_exprs_and_uses):
  #Now do fusion of MiSets after presum because C loads are <<<<<< A,B and Presum loads.
  #We can pass output using both Accumulator and Shared Memory if there is enough memory.
  #However, for FP32 not enough shared memory is available 128*256*4 = 128KB per TB
  #When we fuse two MiSets of size N, then each Mi can use 128/N registers to compute results.
  #Thus, we can only fuse MiSets that have same size 
  #If the first Miset in the order does presum, then we cannot fuse it with those that require presum.

  #Fusion algorithm: 
  # 1. Starting from the most common expr, fuse all MiSets that computes this common expr.
  # 2. For this common expr, fuse computation of this common expr with all uses of this common expr.
  # Fusing a node or a fused set with another fused set is valid when:
  # (i)  No cycles are generated when adding a node to a fused set.
  # (ii) An miset with presum store is not fused with an miset that loads this presum.
  # (iii) Maximum out edges within a fused set should be less than number of fusion slots, 
  #       i.e., fusion should not need a global memory store for postsum and global memory load for postsum.

  FusionSlots = {"Registers": 1, "SharedMemory": 1}
  miset_to_fused = {m: set([m]) for m in partitioning_info.misetnodes}
  if len(partitioning_info.misetnodes) == 1:
    return

  #TODO: Fuse partitioning_info.misetnodes_wo_presum

  if len(partitioning_info.misetnodes_wo_presum()) > 0:
    presum_miset = partitioning_info.misetnodes_wo_presum()[0]
  else:
    presum_miset = None

  for common_expr in sorted_common_exprs_and_uses:
      common_expr_def = None
      #Find miset that stores this common expr
      for miset in partitioning_info.misetnodes:
        if miset.get_store(common_expr) != None:
          common_expr_def = miset
          break

      #Traverse the miset graph backward and fuse nodes
      q = [common_expr_def]
      root = miset
      while len(q) > 0:
        parent = q.pop(0)

        for load in parent.loads:
          child = partitioning_info.misetnodes[load.index]
          fusion_is_valid = len(miset_to_fused[parent]) + len(miset_to_fused[child]) <= MAX_FUSED
          fusion_is_valid = fusion_is_valid and len(parent.miset) == len(child.miset)
          fusion_is_valid = fusion_is_valid and child not in partitioning_info.misetnodes_wo_presum()
          if fusion_is_valid:
            for child_miset in miset_to_fused[child]:
              miset_to_fused[parent].add(child_miset)
              miset_to_fused[child_miset] = miset_to_fused[parent]
          q.append(child)

      common_expr_uses = []
      for miset in partitioning_info.misetnodes:
        if miset.get_load(common_expr) != None:
          common_expr_uses += [miset]

      #Fuse final miset of common expr with its uses
      for miset_load in common_expr_uses:
        fusion_is_valid = len(miset_to_fused[miset]) + len(miset_to_fused[miset_load]) <= MAX_FUSED
        fusion_is_valid = fusion_is_valid and len(miset.miset) == len(miset_load.miset)
        if fusion_is_valid:
          for child_miset in miset_to_fused[miset_load]:
            miset_to_fused[miset].add(child_miset)
            miset_to_fused[child_miset] = miset_to_fused[miset]

  fused_mi_sets = []
  for mi,fused_set in miset_to_fused.items():
    if fused_set not in fused_mi_sets:
      fused_mi_sets += [fused_set]

  fused_mi_sets = [FusedMiSet(partitioning_info.misetnodes, f) for f in fused_mi_sets]
  partitioning_info.set_fused_mi_sets(fused_mi_sets)

  #Generate a DFS based Topological sort order within each fused_mi_set, so that the next immediate load of a store
  #can be passed through registers, while others can be passed through shared memory.
  for fused_mi_set in fused_mi_sets:
    linear_order = fused_mi_set.dfs_topo_order()
    if FusionSlots["Registers"] > 0:
      for idx,_ in enumerate(linear_order):
        if idx == len(linear_order) - 1: continue
        miset1 = linear_order[idx]
        miset2 = linear_order[idx+1]

        for l in miset2.loads:
          store = miset1.get_store(l.expr)
          if store != None:
            store.mem_types += ["reg"]
            l.mem_types += ["reg"]

    #Assign shared memory
    if FusionSlots["SharedMemory"] > 0:
      sh_mem_bufs = [{"buff":b, "status":"avail"} for b in range(FusionSlots["SharedMemory"])]
      stores_live_range = fused_mi_set.live_ranges(linear_order)

      for store_with_start_end in stores_live_range:
        store = store_with_start_end[0]
        start = store_with_start_end[1][0]
        end = store_with_start_end[1][1]

        #If current store.start is more than sh_mem_buff's status 
        #then free sh_mem_buff
        for sh_mem_buff in sh_mem_bufs:
          if sh_mem_buff["status"] != "avail" and sh_mem_buff["status"][1] > end:
            sh_mem_buff["status"] = "avail"

        #If a shared memory buff is available then assign to it
        for sh_mem_buff in sh_mem_bufs:
          if sh_mem_buff["status"] == "avail" and end > start + 1:
            sh_mem_buff["status"] = (start, end)
            buff = f"sh_{sh_mem_buff['buff']}"
            store.mem_types += [buff]
            for miset in linear_order[start:end+1]:
              for load in miset.loads:
                if load.expr == store.expr:
                  load.mem_types += [buff]
            break

  #Set global memory loads/stores
  for miset in partitioning_info.misetnodes:
    for load in miset.loads:
      if len(load.mem_types) == 0:
        load.mem_types += ["gl"]
        partitioning_info.misetnodes[load.index].get_store(load.expr).mem_types += ["gl"]

  #Assign global memory buffers and Cis to postsum updates

  return

  #Immediate use is always passed through registers.
  #Next edge in the linear order is always passed through registers while an 
  #If an Miset has more than one out-edges then load/store using global/shared memory.
  #TODO: how to add shared memory?
  fused_misets = []
  created_fused_misets = []
  for miset in order:
      if miset_to_fused[miset] != [] and miset_to_fused[miset] not in fused_misets:
          if miset_to_fused[miset] in created_fused_misets:
              continue
          created_fused_misets += [miset_to_fused[miset]]
          fused_misets += [FusedMiSet(misetnodes, miset_to_fused[miset])]

  # MiSet.small_print = True
  # print(715, fused_misets)
  for fused in fused_misets:
      for miset in fused:
          store_gl = False
          use_sh = False
          use_reg = False

          all_uses_in_fuse = True
          uses_in_fuse = []
          uses_out_of_fuse = []

          for use in miset_uses[misetnodes.index(miset)]:
              if use not in fused:
                  uses_out_of_fuse += [use]
              else:
                  uses_in_fuse     += [use]
          # print(768, miset, miset_uses[misetnodes.index(miset)])
          for use_in_fuse in uses_in_fuse:
              if order.index(use_in_fuse) == order.index(miset) + 1:
                  #if use is in the fuse set and immediate next in the order
                  for l in use_in_fuse.loads:
                      store = miset.get_store(l.expr)
                      if store != None:
                          l.mem_types = ["reg"]
                          store.mem_types += ["reg"]
                          # print(776, l, store)
              else:
                  #if user is not immediately next then store in global memory
                  for l in use_in_fuse.loads:
                      store = miset.get_store(l.expr)
                      if store != None:
                          l.mem_types = ["gl"]
                          store.mem_types += ["gl"]

          #for stores used outside of fuse set use gl
          for use in uses_out_of_fuse:
              for l in use.loads:
                  store = miset.get_store(l.expr)
                  if store != None:
                      store.mem_types += ["gl"]
                      l.mem_types = ["gl"]

def get_gmem_loads(misetnodes, presums_todo, Mis, M,N,K, tm,tn, rm,rn):
  a_gmem_loads = b_gmem_loads = c_gmem_loads = 0
  all_shmem_loads = 0
  thread_blocks = 0
  
  loads_for_presum = {"a": set(), "b": set()}
  for aorb in ["a", "b"]:
    if len(presums_todo[aorb]) == 0: continue 
    base_matrices = [s[0].get_base_matrices() for s in presums_todo[aorb]]
    base_matrices = reduce(lambda x, y: x.union(y), base_matrices)
    loads_for_presum[aorb].update(base_matrices)

  if len(loads_for_presum["a"]) > 0:
      a_gmem_loads += len(loads_for_presum["a"]) * (M//2) * (K//2) #Both loads and stores
  if len(loads_for_presum["b"]) > 0:
      b_gmem_loads += len(loads_for_presum["b"]) * (N//2) * (K//2) #Both loads and stores

  for partidx, part in enumerate(misetnodes):
    loads_using_presum = {"a": set(), "b": set()}
    loads_without_presum = {"a": set(), "b": set()}

    for mi in part.miset:
      for aorb in ["a", "b"]:
        q = [mi.op1 if aorb == "a" else mi.op2]
        while len(q) > 0:
          expected_load = q.pop(0)
          l = [load for load in presums_todo[aorb] if expected_load == load[0]]

          if len(l) > 0:
              loads_using_presum[aorb].add(expected_load)
          elif isinstance(expected_load, BinOp):
              q.append(expected_load.op1)
              q.append(expected_load.op2)
          elif isinstance(expected_load, Matrix):
              loads_without_presum[aorb].add(expected_load)
  
    m_tiles = (M//tm)//2 * (N//tn)//2
    if len(part.miset) == 1:
        rm_fac = 1
        rn_fac = 1
    elif len(part.miset) == 2:
        rm_fac = 2
        rn_fac = 1
    elif len(part.miset) == 4:
        rm_fac = 2
        rn_fac = 2

    if len(part.miset) == len(Mis):
        shmem_loads = 7/2*tm//rm * tn//rn * (rm + rn)
        rm_fac = 1
        rn_fac = 1
    else:
        shmem_loads = rm_fac*rn_fac*tm//rm * tn//rn * (rm//rm_fac + rn//rn_fac)

    all_shmem_loads = max(all_shmem_loads, shmem_loads)
    a_gmem_loads += rm_fac * rn_fac * m_tiles * (K//2) * ((len(loads_using_presum["a"]) + len(loads_without_presum["a"])) * tm//rm_fac)
    b_gmem_loads += rm_fac * rn_fac * m_tiles * (K//2) * ((len(loads_using_presum["a"]) + len(loads_without_presum["b"])) * tn//rn_fac)
    thread_blocks += len(part.miset) if len(part.miset) < len(Mis) else 4
  return a_gmem_loads, b_gmem_loads, c_gmem_loads

def gen_schedule(C, Ms, m, n, k, tile, regtile):
  all_valid_partitionings = []
  num_all_partitionings = 0
  num_all_valid_partitionings = 0

  sorted_common_exprs = find_times_each_expr_is_used(C, Ms)
  mis_wo_presum = set([Ms[0], Ms[1]])

  for partitioning in list(get_all_set_partitions(Ms)):
    if str(partitioning) != "[[m0], [m1], [m2], [m3], [m4], [m5], [m6]]": continue
    num_all_partitionings += 1
    if not all_miset_of_powerto(partitioning, len(Ms)): continue
    if MAX_PRESUM_BUFFERS > 0 and not atleast_one_miset_wo_presum_is_not_fused(partitioning, mis_wo_presum): continue
    misetnodes = [MiSet(p) for p in partitioning]

    expr_def_by_miset,common_exprs = set_def_of_postsum_exprs(C.parts, partitioning, misetnodes, list(sorted_common_exprs))
    if misetnodes_has_cycle(misetnodes): continue

    presums_todo = apply_presum(misetnodes)
    a_gmem_loads, b_gmem_loads, c_gmem_loads = get_gmem_loads(misetnodes, presums_todo,
                                                              Ms, m,n,k,
                                                              tile[0],tile[1],
                                                              regtile[0],regtile[1])
    all_valid_partitionings += [Partitioning(common_exprs, misetnodes, expr_def_by_miset, presums_todo,
                                             a_gmem_loads, b_gmem_loads, c_gmem_loads)]

  print(f"Valid {len(all_valid_partitionings)} partitionings out of total {num_all_partitionings}")
  sorted_partitionings = sorted(all_valid_partitionings, key=lambda p: p.total_gmem_loads())
  for p in sorted_partitionings[:5]:
    p.init_misetnodes_wo_presum(mis_wo_presum)
    set_presum_computation(p, mis_wo_presum)
    postsum_fusion(p, sorted_common_exprs)
    order = order_Mis(C.parts, p)
    print(p)
    p.print_miset_in_order(order)
    print("\n")

def parse_args():
  parser = argparse.ArgumentParser(description="Generate schedule for Strassen variants")
  parser.add_argument("--m", type=int, default=1024, help="Rows of matrix A/C")
  parser.add_argument("--n", type=int, default=1024, help="Columns of matrix B/C")
  parser.add_argument("--k", type=int, default=1024, help="Columns of matrix A / rows of matrix B")
  parser.add_argument("--algo", type=str, default="strassen_winograd", help="Algorithm name")
  parser.add_argument("--level", type=int, default=1, help="Recursion level")
  parser.add_argument("--datatype", type=str, default="fp32", help="Datatype (for metadata/planning)")
  parser.add_argument("--output_dir", type=str, default=".", help="Output directory for generated artifacts")
  return parser.parse_args()

def _to_posix_relpath(from_dir, to_path):
  rel = os.path.relpath(to_path, from_dir)
  return rel.replace(os.sep, "/")

def _make_rewritten_makefile_content(lines, makefile_dir, repo_root):
  replacements = {
      "CUTLASS": _to_posix_relpath(makefile_dir, os.path.join(repo_root, "cutlass", "include")) + "/",
      "CUTLASS_COMMON": _to_posix_relpath(makefile_dir, os.path.join(repo_root, "cutlass", "examples", "common")),
      "UTIL_INCLUDE": _to_posix_relpath(makefile_dir, os.path.join(repo_root, "cutlass", "tools", "util", "include")) + "/",
      "STRASSEN_CUTLASS": _to_posix_relpath(makefile_dir, os.path.join(repo_root, "include")),
  }

  updated_lines = []
  for line in lines:
    replaced = False
    for key, value in replacements.items():
      if line.startswith(f"{key} ="):
        updated_lines.append(f"{key} = {value}\n")
        replaced = True
        break
    if not replaced:
      updated_lines.append(line)

  return "".join(updated_lines)

def _rewrite_makefile_paths(makefile_path, repo_root):
  makefile_dir = os.path.dirname(makefile_path)

  with open(makefile_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

  rewritten_content = _make_rewritten_makefile_content(lines, makefile_dir, repo_root)

  with open(makefile_path, "w", encoding="utf-8") as f:
    f.write(rewritten_content)

def _files_have_same_content(path1, path2):
  if not (os.path.exists(path1) and os.path.exists(path2)):
    return False
  with open(path1, "rb") as f1, open(path2, "rb") as f2:
    return f1.read() == f2.read()

def copy_example_files_for_datatype(datatype, output_dir):
  dtype = datatype.lower()
  if dtype in ["fp32", "f32"]:
    src_rels = [
      os.path.join("example", "strassen_gemm"),
      os.path.join("example", "00_basic_gemm"),
    ]
  elif dtype in ["fp16", "f16"]:
    src_rels = [
      os.path.join("example", "strassen_gemm_ampere_f16_tensorop"),
      os.path.join("example", "14_ampere_f16_tensorop_gemm"),
    ]
  else:
    raise ValueError(f"Unsupported datatype '{datatype}'. Supported: f32, f16")

  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  copied = 0
  skipped_unchanged = 0
  for src_rel in src_rels:
    src_dir = os.path.join(repo_root, src_rel)
    if not os.path.isdir(src_dir):
      raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    for root, _, files in os.walk(src_dir):
      for filename in files:
        is_makefile = filename == "Makefile"
        is_cuda = filename.endswith(".cu") or filename.endswith(".cuh")
        if not (is_makefile or is_cuda):
          continue

        rel_path = os.path.relpath(os.path.join(root, filename), src_dir)
        dst_path = os.path.join(output_dir, os.path.basename(src_rel), rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        src_path = os.path.join(root, filename)

        if is_makefile:
          with open(src_path, "r", encoding="utf-8") as f:
            src_lines = f.readlines()
          expected_content = _make_rewritten_makefile_content(
            src_lines,
            os.path.dirname(dst_path),
            repo_root,
          )

          if os.path.exists(dst_path):
            with open(dst_path, "r", encoding="utf-8") as f:
              if f.read() == expected_content:
                skipped_unchanged += 1
                continue

          with open(dst_path, "w", encoding="utf-8") as f:
            f.write(expected_content)
          copied += 1
        else:
          if _files_have_same_content(src_path, dst_path):
            skipped_unchanged += 1
            continue
          shutil.copy2(src_path, dst_path)
          copied += 1

  print(f"Copied {copied} file(s), skipped {skipped_unchanged} unchanged file(s) from {', '.join(src_rels)} to {output_dir}")

def build(m,n,k,algo,level,datatype,output_dir):
    if algo != "strassen_winograd":
      raise ValueError(f"Unsupported algo '{algo}'. Supported: strassen_winograd")
    if level > 2:
      raise ValueError(f"Unsupported level '{level}'. Current implementation supports level <= 2")

    os.makedirs(output_dir, exist_ok=True)
    copy_example_files_for_datatype(datatype, output_dir)
    # if algo == "strassen_winograd":

if __name__ == "__main__":
    args = parse_args()
    build(args.m, args.n, args.k, args.algo, args.level, args.datatype, args.output_dir)

    c, mis = strassen_winograd(m, n, k)

    gen_schedule(c, mis, m, n, k, (256, 128, 8), (16, 8))