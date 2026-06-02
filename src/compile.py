import collections
import argparse
import math
import os
import shutil

# Mocking hardware specs for the cost model/scheduler
HW_SPECS = {
    "A100": {"sm_count": 108, "shared_mem_per_sm": 168 * 1024, "regs_per_thread": 255},
    "H200": {"sm_count": 132, "shared_mem_per_sm": 228 * 1024, "regs_per_thread": 255}
}


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


def _files_have_same_content(path1, path2):
    if not (os.path.exists(path1) and os.path.exists(path2)):
        return False
    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        return f1.read() == f2.read()


def copy_example_files_for_datatype(datatype, output_dir, dry_run=False):
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
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    copied = 0
    skipped_unchanged = 0
    planned = 0
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
                if not dry_run:
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

                    if dry_run:
                        planned += 1
                        print(f"[DRY-RUN] Would write: {dst_path}")
                    else:
                        with open(dst_path, "w", encoding="utf-8") as f:
                            f.write(expected_content)
                        copied += 1
                else:
                    if _files_have_same_content(src_path, dst_path):
                        skipped_unchanged += 1
                        continue
                    if dry_run:
                        planned += 1
                        print(f"[DRY-RUN] Would copy: {src_path} -> {dst_path}")
                    else:
                        shutil.copy2(src_path, dst_path)
                        copied += 1

    if dry_run:
        print(
            f"[DRY-RUN] Planned {planned} file write/copy operation(s), "
            f"skipped {skipped_unchanged} unchanged file(s) "
            f"from {', '.join(src_rels)} to {output_dir}"
        )
    else:
        print(
            f"Copied {copied} file(s), skipped {skipped_unchanged} unchanged file(s) "
            f"from {', '.join(src_rels)} to {output_dir}"
        )

class StrassenCompiler:
    KernelInfoFP32 = {
        "K2": {"DataType": "fp32", "TileSize" : (128, 128), "SharedMemoryBytes": 64*1024},
        "K1": {"DataType": "fp32", "TileSize" : (256, 128), "SharedMemoryBytes": 64*1024},
        "K4": {"DataType": "fp32", "TileSize" : (256, 128), "SharedMemoryBytes": 64*1024},
    }
    KernelInfoFP16 = {
        "K2": {"DataType": "fp32", "TileSize" : (128, 128), "SharedMemoryBytes": 64*1024},
        "K1": {"DataType": "fp32", "TileSize" : (256, 128), "SharedMemoryBytes": 96*1024},
        "K4": {"DataType": "fp32", "TileSize" : (256, 128), "SharedMemoryBytes": 96*1024},
    }
    def __init__(self, hw_name="A100", print_all_recursive=False):
        self.hw = HW_SPECS.get(hw_name)
        self.memo = {}
        self.fusion_memo = {}
        self.last_fusion_schedule = None
        self.print_all_recursive = print_all_recursive

    def _trace_schedule(self, level, mnk, src_m, schedule, from_memo=False):
        """Print per-call schedules when recursive tracing is enabled."""
        if not self.print_all_recursive:
            return
        src_label = src_m if src_m is not None else "Base"
        memo_tag = " [memo]" if from_memo else ""
        print(
            f"[Trace] level={level}, mnk={mnk}, src={src_label}{memo_tag} -> {schedule}"
        )

    def build_output_sum_dag(self, root_node):
        """
        Constructs the DAG of sub-matmuls and their dependencies.
        Nodes are sub-matmuls (M_i), edges represent output-sum dependencies.
        """
        dag = collections.defaultdict(list)
        matmuls = []

        # Collect all "@" BinOp nodes from any root expression type (e.g., Combine4x4).
        def collect_matmuls(node, *_):
            if getattr(node, "op", None) == "@":
                matmuls.append(node)
                # A matmul is a leaf for scheduling; no need to traverse deeper.
                return False
            return True

        if hasattr(root_node, "visit"):
            root_node.visit(collect_matmuls)

        # De-duplicate while preserving discovery order.
        unique_matmuls = list(dict.fromkeys(matmuls))

        # Traverse from output back to matmuls to find common expressions.
        # Simplified for this prototype: only return discovered matmuls.
        return dag, unique_matmuls

    def get_min_input_sum_loads(self, sub_matmuls):
        """
        Heuristic to find the sub-matmul that requires the least 
        additional quarter-matrix loads for its input-sums[cite: 415, 780].
        """
        # In Strassen-Winograd, M0 (A0*B0) requires 0 additional loads.
        return sub_matmuls[0] 

    def estimate_cost(self, schedule, mnk):
        """
        Simulates RUNORGETCACHED[cite: 1102].
        In a real compiler, this would run a profiler or use a performance model.
        """
        # Logic: K2 is faster for smaller sizes; K1/K4 faster for larger sizes [cite: 1768, 1769]
        m, n, k = mnk
        if "K2" in str(schedule) and m < 2048:
            return m * n * k * 0.8 # Boost K2 for small matrices
        return m * n * k * 1.0

    def should_use_k2(self, mnk):
        """
        Level-1 policy requested by user:
        - For strictly smaller than 2048^3, prefer all-in-one K2.
        - Otherwise, prefer K1 + 6*K4 decomposition.
        """
        return all(dim < 2048 for dim in mnk)

    def count_all_schedules(self, level):
        """Count all schedule combinations for a given recursion level."""
        if level <= 0:
            return 0
        if level == 1:
            return 2
        return self.count_all_schedules(level - 1) ** 6

    def iter_all_schedules(self, alg_func, level, mnk, src_m=None):
        """Yield every possible schedule instead of selecting a single best one."""
        root_c, sub_matmuls = alg_func(*mnk)
        _, mis = self.build_output_sum_dag(root_c)

        in_sum_m = self.get_min_input_sum_loads(sub_matmuls)
        split_sched = [f"K1({in_sum_m})"]
        split_sched += [f"K4({m})" for m in mis if m != in_sum_m]
        k2_sched = [f"K2({src_m if src_m else 'Base'})"]

        if level == 1:
            # Exactly 2 schedules at level-1.
            yield k2_sched
            yield split_sched
            return

        child_ms = [mi for mi in mis if mi != in_sum_m]
        half_mnk = (mnk[0] // 2, mnk[1] // 2, mnk[2] // 2)

        def _build_children(idx, chosen):
            if idx == len(child_ms):
                yield list(chosen)
                return

            m = child_ms[idx]
            label = f"K4({m})"
            for child_sched in self.iter_all_schedules(
                alg_func, level - 1, half_mnk, m
            ):
                chosen.append({label: child_sched})
                yield from _build_children(idx + 1, chosen)
                chosen.pop()

        # For level>1, each of 6 K4 branches can independently choose from child schedules.
        for child_combo in _build_children(0, []):
            yield [f"K1({in_sum_m})"] + child_combo

    def output_sum_fusion(self, dag, schedule, mnk, kernel_type):
        """Algorithm-2 style greedy output-sum fusion over a 1-level DAG.

        Only `K2` or `K4` are fusable. `K1` is never fused.
        """

        def _collect_targets(items):
            targets = []
            for item in items:
                if isinstance(item, str) and item.startswith(f"{kernel_type}("):
                    targets.append(item)
                elif isinstance(item, dict):
                    for key in item.keys():
                        if key.startswith(f"{kernel_type}("):
                            targets.append(key)
            return targets

        def _kernel_token(label):
            start = label.find("(")
            end = label.rfind(")")
            if start == -1 or end == -1 or end <= start:
                return label
            return label[start + 1:end]

        def _build_target_dag(targets, src_dag):
            # Try to project edges from the expression DAG to scheduled kernels.
            # If no edge can be projected, keep a deterministic linear fallback.
            target_tokens = {_kernel_token(t): t for t in targets}
            target_adj = {t: [] for t in targets}
            added = 0

            for parent, users in src_dag.items():
                p_txt = str(parent)
                p_label = None
                for token, label in target_tokens.items():
                    if token in p_txt:
                        p_label = label
                        break
                if p_label is None:
                    continue

                for user in users:
                    u_txt = str(user)
                    for token, label in target_tokens.items():
                        if token in u_txt and label != p_label and label not in target_adj[p_label]:
                            target_adj[p_label].append(label)
                            added += 1
                            break

            if added == 0:
                for idx in range(len(targets) - 1):
                    target_adj[targets[idx]].append(targets[idx + 1])

            return target_adj

        def _bfs_order(adj):
            indeg = {n: 0 for n in adj.keys()}
            for parent, children in adj.items():
                for child in children:
                    if child in indeg:
                        indeg[child] += 1

            q = collections.deque([n for n, d in indeg.items() if d == 0])
            order = []
            visited = set()
            while q:
                node = q.popleft()
                if node in visited:
                    continue
                visited.add(node)
                order.append(node)
                for child in adj.get(node, []):
                    if child not in indeg:
                        continue
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        q.append(child)

            for node in adj.keys():
                if node not in visited:
                    order.append(node)
            return order

        def _has_path(adj, src, dst):
            if src == dst:
                return True
            seen = set([src])
            q = collections.deque([src])
            while q:
                node = q.popleft()
                for nxt in adj.get(node, []):
                    if nxt == dst:
                        return True
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            return False

        def _build_group_graph(fused_map, adj):
            group_adj = {}
            for node, group in fused_map.items():
                rep = tuple(sorted(group))
                if rep not in group_adj:
                    group_adj[rep] = set()

            for parent, children in adj.items():
                if parent not in fused_map:
                    continue
                parent_rep = tuple(sorted(fused_map[parent]))
                for child in children:
                    if child not in fused_map:
                        continue
                    child_rep = tuple(sorted(fused_map[child]))
                    if parent_rep != child_rep:
                        group_adj[parent_rep].add(child_rep)

            return {k: list(v) for k, v in group_adj.items()}

        def _thread_blocks_for_kernel(kernel, problem):
            m, n, _ = problem
            if kernel == "K4":
                m = max(1, m // 2)
                n = max(1, n // 2)
            tile_m, tile_n = self.KernelInfoFP32[kernel]["TileSize"]
            return math.ceil(m / tile_m) * math.ceil(n / tile_n)

        if kernel_type not in {"K2", "K4"}:
            return {
                "kernel": kernel_type,
                "fused_map": {},
                "groups": [],
                "notes": "Unsupported kernel type for fusion",
            }

        targets = _collect_targets(schedule)
        if not targets:
            return {
                "kernel": kernel_type,
                "fused_map": {},
                "groups": [],
                "notes": f"No {kernel_type} kernels available for fusion",
            }

        target_dag = _build_target_dag(targets, dag)
        fused_map = {t: set([t]) for t in targets}

        max_fused = 4
        fusion_buffers = 2
        sm_count = self.hw["sm_count"] if self.hw else 1
        tb_per_kernel = _thread_blocks_for_kernel(kernel_type, mnk)

        changed = True
        while changed:
            changed = False
            for parent in _bfs_order(target_dag):
                if parent not in fused_map:
                    continue

                parent_group = fused_map[parent]
                users = [
                    u
                    for u in target_dag.get(parent, [])
                    if u in fused_map and fused_map[u] is not parent_group
                ]

                for user in users:
                    user_group = fused_map[user]

                    # (i) K1 does input-sum work and is never fusable.
                    if parent.startswith("K1(") or user.startswith("K1("):
                        continue

                    # (ii) Number of fused uses should not exceed fusion buffers.
                    fused_uses = set()
                    for p in parent_group:
                        for u in target_dag.get(p, []):
                            if u in fused_map and fused_map[u] is parent_group:
                                fused_uses.add(u)
                    if len(fused_uses) >= fusion_buffers:
                        continue

                    # (iii) Keep fused code size bounded.
                    merged_group = parent_group.union(user_group)
                    if len(merged_group) > max_fused:
                        continue

                    # (iv) Merging should not create a dependency cycle.
                    group_graph = _build_group_graph(fused_map, target_dag)
                    parent_rep = tuple(sorted(parent_group))
                    user_rep = tuple(sorted(user_group))
                    if _has_path(group_graph, user_rep, parent_rep):
                        continue

                    # (v) Keep enough thread-block level parallelism.
                    current_groups = {tuple(sorted(group)) for group in fused_map.values()}
                    projected_group_count = max(1, len(current_groups) - 1)
                    projected_total_tbs = projected_group_count * tb_per_kernel
                    if projected_total_tbs <= sm_count:
                        continue

                    for node in merged_group:
                        fused_map[node] = merged_group
                    changed = True

        unique_groups = []
        for group in fused_map.values():
            grp = sorted(group)
            if grp not in unique_groups:
                unique_groups.append(grp)

        fused_map_serialized = {node: sorted(group) for node, group in fused_map.items()}
        return {
            "kernel": kernel_type,
            "fused_map": fused_map_serialized,
            "groups": unique_groups,
            "mnk": mnk,
            "sm_count": sm_count,
            "tb_per_kernel": tb_per_kernel,
            "dag_edges": sum(len(v) for v in target_dag.values()),
        }

    def gen_sched(self, alg_func, level, mnk, src_m=None):
        """
        Algorithm 1: Generate Kernel Schedule[cite: 1061].
        Recursively finds the optimal combination of K1, K2, K3, K4 kernels.
        """
        memo_key = (level, mnk)
        if memo_key in self.memo:
            cached = self.memo[memo_key]
            self.last_fusion_schedule = self.fusion_memo.get(memo_key)
            self._trace_schedule(level, mnk, src_m, cached, from_memo=True)
            return cached

        root_c, sub_matmuls = alg_func(*mnk)
        dag, mis = self.build_output_sum_dag(root_c)

        def _count_kernel_occurrences(node, kernel_prefix):
            count = 0
            if isinstance(node, str):
                return 1 if node.startswith(f"{kernel_prefix}(") else 0
            if isinstance(node, list):
                for item in node:
                    count += _count_kernel_occurrences(item, kernel_prefix)
                return count
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(key, str) and key.startswith(f"{kernel_prefix}("):
                        count += 1
                    count += _count_kernel_occurrences(value, kernel_prefix)
                return count
            return 0

        if level == 1:
            # Schedule 1: Split Input-Sums and Matmuls (K1 + K4s) [cite: 930]
            in_sum_m = self.get_min_input_sum_loads(sub_matmuls)
            s1_1 = [f"K1({in_sum_m})"]
            s1_1 += [f"K4({m})" for m in mis if m != in_sum_m]
            s1_1_fusion = self.output_sum_fusion(dag, s1_1, mnk, kernel_type="K4")
            
            # Schedule 2: All-in-one tile Strassen (K2) [cite: 930]
            s2_1 = [f"K2({src_m if src_m else 'Base'})"]
            s2_1_fusion = self.output_sum_fusion(dag, s2_1, mnk, kernel_type="K2")

            # Follow explicit size-based policy for level-1 schedule selection.
            # Keep explicit K1/K4 kernels for larger sizes to make recursion visible.
            best_s = s2_1 if self.should_use_k2(mnk) else s1_1
            best_fusion = s2_1_fusion if self.should_use_k2(mnk) else s1_1_fusion
            
            self.memo[memo_key] = best_s
            self.fusion_memo[memo_key] = best_fusion
            self.last_fusion_schedule = best_fusion
            self._trace_schedule(level, mnk, src_m, best_s)
            return best_s

        else:
            # Recursive case for level > 1 
            in_sum_m = self.get_min_input_sum_loads(sub_matmuls)
            # First sub-matmul at this level is handled by K1.
            s_l = [f"K1({in_sum_m})"]
            child_has_multi_k2 = False
            
            for m in [mi for mi in mis if mi != in_sum_m]:
                # Recursively get schedule for sub-problem [cite: 1188]
                child_sched = self.gen_sched(
                    alg_func,
                    level - 1,
                    (mnk[0] // 2, mnk[1] // 2, mnk[2] // 2),
                    m,
                )
                if _count_kernel_occurrences(child_sched, "K2") > 1:
                    child_has_multi_k2 = True
                # Show explicit expansion: this K4 branch becomes a child schedule.
                s_l.append({f"K4({m})": child_sched})

            result = s_l
            fusion_kernel = "K2" if child_has_multi_k2 else "K4"
            result_fusion = self.output_sum_fusion(dag, result, mnk, kernel_type=fusion_kernel)
            self.memo[memo_key] = result
            self.fusion_memo[memo_key] = result_fusion
            self.last_fusion_schedule = result_fusion
            self._trace_schedule(level, mnk, src_m, result)
            return result


def build(
    level,
    m,
    n,
    k,
    datatype,
    output_dir,
    dry_run=False,
    hw="A100",
    print_all_recursive=False,
    print_all_schedules=False,
):
    """Build/copy Strassen artifacts and generate schedules programmatically.

    Args:
        level: Strassen recursion level.
        m: Matrix M dimension.
        n: Matrix N dimension.
        k: Matrix K dimension.
        datatype: Datatype selector for example sources ("fp32"/"f32" or "fp16"/"f16").
        output_dir: Destination directory for copied Makefile/CUDA artifacts.
        dry_run: If True, preview copy/write operations without modifying files.
        hw: Target hardware key from HW_SPECS.
        print_all_recursive: If True, print schedule at each recursive call.
        print_all_schedules: If True, enumerate and print all schedules (also implies dry-run for file copy).

    Returns:
        list: Final schedule in normal mode, or a list of all schedules when
            print_all_schedules=True.

    Example:
        from src.compile import build

        final_schedule = build(
            level=1,
            m=4096,
            n=4096,
            k=4096,
            datatype="fp32",
            output_dir="out",
            dry_run=True,
            hw="A100",
            print_all_recursive=False,
            print_all_schedules=False,
        )
    """
    effective_dry_run = dry_run or print_all_schedules

    compiler = StrassenCompiler(
        hw_name=hw,
        print_all_recursive=print_all_recursive,
    )
    from algos.strassen_winograd_level1 import strassen_winograd

    copy_example_files_for_datatype(datatype, output_dir, dry_run=effective_dry_run)

    if print_all_schedules:
        total = compiler.count_all_schedules(level)
        print(f"Total schedules at level {level}: {total}")
        schedules = []
        try:
            for idx, schedule in enumerate(
                compiler.iter_all_schedules(
                    strassen_winograd,
                    level=level,
                    mnk=(m, n, k),
                ),
                start=1,
            ):
                print(f"Schedule {idx}/{total}: {schedule}")
                fusion_kernel = "K2" if "K2(" in str(schedule) else "K4"
                fusion_schedule = compiler.output_sum_fusion(
                    dag={},
                    schedule=schedule,
                    mnk=(m, n, k),
                    kernel_type=fusion_kernel,
                )
                print(f"Fusion Schedule {idx}/{total}: {fusion_schedule}")
                schedules.append(schedule)
        except BrokenPipeError:
            pass
        return schedules

    final_schedule = compiler.gen_sched(
        strassen_winograd,
        level=level,
        mnk=(m, n, k),
    )
    print(f"Generated Optimal Schedule:\n{final_schedule}")
    print(f"Generated Fusion Schedule:\n{compiler.last_fusion_schedule}")
    return final_schedule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Strassen kernel schedules")
    parser.add_argument("--level", type=int, required=True, help="Strassen recursion level")
    parser.add_argument("--m", type=int, required=True, help="Matrix M dimension")
    parser.add_argument("--n", type=int, required=True, help="Matrix N dimension")
    parser.add_argument("--k", type=int, required=True, help="Matrix K dimension")
    parser.add_argument("--datatype", type=str, required=True, help="Datatype for selecting example sources (fp32/fp16)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for copied example artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Preview copy/write operations without modifying files")
    parser.add_argument("--hw", type=str, default="A100", choices=sorted(HW_SPECS.keys()), help="Target hardware")
    parser.add_argument(
        "--print-all-recursive",
        action="store_true",
        help="Print schedule generated at each recursive call",
    )
    parser.add_argument(
        "--print-all-schedules",
        action="store_true",
        help="Print all possible schedules for the requested level",
    )
    args = parser.parse_args()
    build(
        level=args.level,
        m=args.m,
        n=args.n,
        k=args.k,
        datatype=args.datatype,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        hw=args.hw,
        print_all_recursive=args.print_all_recursive,
        print_all_schedules=args.print_all_schedules,
    )