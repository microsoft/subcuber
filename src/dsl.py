from functools import reduce

class Matrix:
  def __init__(self, shape, start=(0,0), parent=None):
    self.shape = shape
    
    if parent == None:
        parent_start = (0,0)
    else:
        parent_start = parent.start

    self.start = (parent_start[0] + start[0], parent_start[1] + start[1])
    self.name = ""
    self.parent = parent
    #self.matmul_def_by = None
    self.presum_used_by = []

  def __getitem__(self, val):
    shape = []
    assert (len(val) == 2)
    for d,sl in zip(self.shape, val):
        stop = d if sl.stop == None else sl.stop
        start = 0 if sl.start == None else sl.start
        shape += [stop-start]

    return Matrix(shape, start=[ 0 if sl.start == None else sl.start for sl in val], parent=self)

  def __add__(self, m2):
    return BinOp(self, m2, "+")
  
  def __sub__(self, m2):
    return BinOp(self, m2, "-")
  
  def __matmul__(self, m2):
    return BinOp(self, m2, "@")

  def var_names(self, frame_locals):
    for name,var in frame_locals.items():
      if isinstance(var,Matrix):
        var.name = name 

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

  def visit(self, visitor, *visitor_args):
    if visitor(self, visitor_args):
      if self.parent is not None:
        self.parent.visit(visitor, visitor_args)

  def print_tree(self, visited=set()):
    if self in visited:
      return
    visited.add(self)
    if isinstance(self, BinOp):
      self.op1.print_tree(visited)
      self.op2.print_tree(visited)
      print(f"{self.name} = {self.op1.name} {self.op} {self.op2.name}")
    elif isinstance(self, Combine4x4):
      out = []
      for part in self.parts:
        part.print_tree(visited)
        out += [part.name]
      print(f"{self.name} = {out}")
    elif isinstance(self, Matrix):
      out = f"{self.name} = "
      if self.parent is not None:
        out += f"{self.parent.name}[{self.start[0]}:{self.shape[0]+self.start[0]}, {self.start[1]}:{self.shape[1]+self.start[1]}]"
      else:
        out += f"Matrix({self.shape[0]}, {self.shape[1]})"
      print(out)
  
  def get_base_matrices(self, stopFn = None):
    if stopFn is not None and stopFn(self) == True:
      return set([self])

    if isinstance(self, BinOp):
      return self.op1.get_base_matrices(stopFn).union(self.op2.get_base_matrices(stopFn))
    if isinstance(self, Matrix):
      if self.parent is not None and self.parent.parent is None:
        return set([self] )
      if self.parent is not None:
        return self.parent.get_base_matrices(stopFn)
      if self.parent is None:
        return set()

class BinOp(Matrix):
    def __init__(self, m1, m2, op):
      super().__init__((m1.shape[0], m2.shape[1]))
      self.op = op
      self.op1 = m1
      self.op2 = m2
    
    def get_base_matrices_op1(self):
      return self.op1.get_base_matrices()
    
    def get_base_matrices_op2(self):
      return self.op2.get_base_matrices()

    def get_matmul_matrices(self):
      return set(self.get_base_matrices(stopFn = lambda binop: binop.op == "@"))

    def get_all_children(self):
      all_children = set()
      def children_visit(node, *args):
        all_children.add(node)
        return True
      self.visit(children_visit)
      return all_children
    
    def get_children_at_levels(self):
      result = {}
      q = [(self,0)]
      while len(q) >= 1:
        node, level = q.popleft()
        if level not in result:
          result[level] = []
        result[level] += [node]
        if isinstance(node, BinOp):
          q.append((node.op1, level+1))
          q.append((node.op2, level+1))

    def visit(self, visitor, *visitor_args):
      if (visitor(self, visitor_args)):
        self.op1.visit(visitor, visitor_args)
        self.op2.visit(visitor, visitor_args)

class Combine4x4(Matrix):
  def __init__(self, parts):
    super().__init__((parts[0].shape[0]*2, parts[0].shape[1]*2))
    self.parts = parts

  def visit(self, visitor, *visitor_args):
    if (visitor(self, visitor_args)):
      for p in self.parts:
        p.visit(visitor, visitor_args)


class MiSetLoadStore:
  def __init__(self, expr, misetindex):
    self.expr = expr
    self.index = misetindex
    #For load only one mem type but for store multiple mem types
    self.mem_types = []

  def __repr__(self):
    return "("+repr(self.expr) + ", " + repr(self.index) + ", " + repr(self.mem_types) + ")"

  def __str__(self):
    return repr(self)

class MiSet:
  small_print = False

  def __init__(self, miset):
    self.miset = miset
    self.stores = [] #A tuple of (expr, reg/gl/sh)
    self.loads = [] #A tuple of (loaded expr, miset's index, reg/gl/sh)
    self.uses_stores = [] #A tuple of (expr, miset)
    self.common_exprs = []
    self.presum_stores = {"a":[], "b": []}
    self.presum_loads = {"a": [], "b": []}

  def add_store(self, store):
    if self.get_store(store) == None:
      self.stores += [MiSetLoadStore(store, 0)]

  def get_store(self, expr):
    for store in self.stores:
      if expr == store.expr:
        return store
      
    return None

  def add_load(self, load, src):
    if self.get_load(load,src) == None:
      self.loads += [MiSetLoadStore(load, src)]
  
  def get_load(self, load, src = -1):
    for l in self.loads:
      if l.expr == load and (src == -1 or l.index == src):
        return l
    return None
          
  def add_presum_load(self, aorb, expr, source):
    for pl in self.presum_loads[aorb]:
      if pl[0] == expr:
        return
    self.presum_loads[aorb] += [(expr, source)]

  def __repr__(self):
    if MiSet.small_print:
      return "{" + repr(self.miset) + "}"
    return "{" + repr(self.miset) + ", loads: " + repr(self.loads) + ", stores: " + repr(self.stores) + " || loads: " + repr(self.presum_loads) + ", stores: " + repr(self.presum_stores)+"}"
  
  def __str__(self):
    return repr(self)

  def full_str(self, indent = 0):
    return repr(self.miset) + "\n" + \
           " "*indent + "post loads: " + repr(self.loads) + \
           " "*indent + "stores: " + repr(self.stores) + "\n" + \
           " "*indent + "pre loads: " + repr(self.presum_loads) + \
           " "*indent + "stores: " + repr(self.presum_stores)

class FusedMiSet:
  def __init__(self, misetnodes, misets):
    self.misets = []
    self.stores = []
    self.loads = []
    self.fused_loads = []
    self.fused_stores = []

    self.misets = misets
    for store in reduce(lambda x, y: x+y, [miset.stores for miset in self.misets]):
      store_only_in_fused = True
      store_is_used_in_fused = False
      #If a store is used by a node, which is not in the fused set
      #then add the store in self.stores
      for miset2 in misetnodes:
        load_this_store = [True for load in miset2.loads \
                            if load.expr == store.expr]
        if store_only_in_fused and len(load_this_store) > 0 and miset2 not in self.misets:
          store_only_in_fused = False
          break
      
      for miset2 in self.misets:
        load_this_store = [True for load in miset2.loads \
                            if load.expr == store.expr]
        if not store_is_used_in_fused and len(load_this_store) > 0 and miset2 in self.misets:
          store_is_used_in_fused = True
          break

      if not store_only_in_fused:
        #Store is used outside of this fused miset
        self.stores += [store]
      if store_is_used_in_fused:
        #Store is used within this fused miset
        self.fused_stores += [store]

    for load in reduce(lambda x, y: x+y, [miset.loads for miset in self.misets]):
      load_in_fused = False

      #If a load is computed (stored) by a node, which is outside of fused set
      #then add the load in self.loads
      if misetnodes[load.index] not in self.misets:
        load_in_fused = False

      if misetnodes[load.index] in self.misets:
        load_in_fused = True

      if load_in_fused:
        #This load is computed (stored) by a node in the fused set
        self.fused_loads += [load]
      else:
        #This load is computed (stored) by a node not in the fused set
        self.loads  += [load]

  def dfs_topo_order(self):
    visited = set()
    stack = []

    def topologicalSortUtil(miset, stack):
      visited.add(miset)

      #First do children where data cannot be passed through registers.
      prefered_children = []
      other_children = []
      for child in miset.uses_stores:
        if len(child[1].stores) == 1:
          prefered_children += [child]
        else:
          other_children += [child]

      for child in (other_children + prefered_children):
        if child[1] not in visited and child[1] in self.misets:
          topologicalSortUtil(child[1], stack)

      # Push current vertex to stack which stores the result
      stack.append(miset)

    #We want an order where data from one miset to its use can be passed through register.
    #This is not possible, if the use's miset has more than one store.
    #Thus, we want misets with more than one store before anyone else in the stack.
    prefered_children = []
    other_children = []
    for miset in self.misets:
      if len(miset.stores) == 1:  
        prefered_children += [miset]
      else:
        other_children += [miset]

    for miset in other_children + prefered_children:
      if miset not in visited:
        topologicalSortUtil(miset, stack)

    return stack[::-1]
  
  def live_ranges(self, linear_order):
    ranges = {s: [-1, -1] for s in self.fused_stores}

    #TODO: below can be faster  
    for store in self.fused_stores:
      for idx,miset in enumerate(linear_order):
        for store1 in miset.stores:
          if store == store1:
            ranges[store][0] = idx
            break

        for load in miset.loads:
          if load.expr == store.expr and load in self.fused_loads:
            ranges[store][1] = idx

    ranges = sorted(list(ranges.items()), key = lambda x: x[1][0])

    return ranges

  def __repr__(self):
    MiSet.small_print = True
    r = "F" + repr(self.misets)
    # r += " " + repr(self.loads)
    # r += " ; "
    # r += repr(self.stores)
    MiSet.small_print = False
    return r

  def full_str(self):
    return "{\n" + "\n".join([miset.full_str(2) for miset in self.dfs_topo_order()]) + "\n}"