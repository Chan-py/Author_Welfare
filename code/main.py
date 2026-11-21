import argparse
import time

from algorithms import METHODS, SEEDED_METHODS
from utils import cache_path, current_memory, load_hypergraph, peak_memory, reset_peak_memory, save_result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. real/icde2024")
    parser.add_argument("--c", type=int, required=True, help="per-author paper cap c")
    parser.add_argument("--method", default="welfare_greedy", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="random seed, used only by random_reject")
    return parser.parse_args()


def main():
    args = get_args()
    edges, node_ids = load_hypergraph(args.dataset)
    seed = args.seed if args.method in SEEDED_METHODS else None
    kwargs = {} if seed is None else {"seed": seed}

    reset_peak_memory()
    base = current_memory()
    start = time.perf_counter()
    retained = METHODS[args.method](edges, len(node_ids), args.c, **kwargs)
    runtime = time.perf_counter() - start
    memory = peak_memory() - base  # extra peak RSS (bytes) used by the algorithm

    path = cache_path(args.dataset, args.c, args.method, seed)
    save_result(path, retained, edges, node_ids, runtime, memory)
    print(f"{path}: {len(retained)}/{len(edges)} papers, {runtime:.3f}s, {memory / 2**20:.1f}MiB")


if __name__ == "__main__":
    main()
