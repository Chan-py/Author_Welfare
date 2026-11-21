import argparse
import csv
import os

import measures
from algorithms import METHODS, SEEDED_METHODS
from utils import cache_path, load_hypergraph, load_result


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. real/icde2024")
    parser.add_argument("--c", type=int, required=True, help="per-author paper cap c")
    parser.add_argument("--method", default="welfare_greedy", choices=list(METHODS))
    parser.add_argument("--seed", type=int, default=0, help="random seed, used only by random_reject")
    parser.add_argument("--out", required=True, help="csv path, e.g. ../results/results.csv")
    return parser.parse_args()


def append_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = get_args()
    edges, node_ids = load_hypergraph(args.dataset)
    seed = args.seed if args.method in SEEDED_METHODS else None
    result = load_result(cache_path(args.dataset, args.c, args.method, seed))
    retained = result["retained"]
    n = len(node_ids)
    ratio = measures.retention_ratio(edges, n, retained)
    welfare = measures.welfare(ratio)
    lp = measures.opt_lp(edges, n, args.c)
    greedy = args.method in ("welfare_greedy", "welfare_greedy_trim")  # U is defined for greedy only

    row = {
        "dataset": args.dataset,
        "c": args.c,
        "method": args.method,
        "seed": "" if seed is None else seed,
        "num_retained_papers": measures.num_retained_papers(retained),
        "welfare": welfare,
        "min_retention_ratio": measures.min_retention_ratio(ratio),
        "num_zero_retention_authors": measures.num_zero_retention_authors(ratio),
        "num_partial_retention_authors": measures.num_partial_retention_authors(ratio),
        "lost_welfare": measures.lost_welfare(ratio),
        "opt_lp": lp,
        "ratio_to_lp": welfare / lp,
        "certificate_u": measures.certificate_u(edges, n, args.c, retained) if greedy else "",
        "retention_q10": measures.retention_quantile(ratio, 0.10),
        "retention_q25": measures.retention_quantile(ratio, 0.25),
        "num_collateral_authors": measures.num_collateral_authors(edges, n, args.c, ratio),
        "loss_share_over_cap": measures.loss_share_over_cap(edges, n, args.c, ratio),
        "retention_q10_over_cap": measures.retention_quantile_over_cap(edges, n, args.c, ratio, 0.10),
        "retention_q25_over_cap": measures.retention_quantile_over_cap(edges, n, args.c, ratio, 0.25),
        **measures.author_group_losses(edges, n, args.c, ratio),
        "runtime": result["runtime"],
        "memory": result["memory"],
    }
    append_row(args.out, row)
    print(row)


if __name__ == "__main__":
    main()
