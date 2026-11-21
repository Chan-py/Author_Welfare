import numpy as np

from function import degrees, over_cap_authors, paper_weights, solve_lp, split_p0


def retention_ratio(edges, n_nodes, retained):
    """rho_a(S) = |S ∩ P(a)| / d_a for every author a."""
    deg = degrees(edges, n_nodes)
    kept = [0] * n_nodes
    for p in retained:
        for a in edges[p]:
            kept[a] += 1

    return [kept[a] / deg[a] for a in range(n_nodes)]


def num_retained_papers(retained):
    return len(retained)


def welfare(ratio):
    return sum(ratio)


def min_retention_ratio(ratio):
    return min(ratio)


def num_zero_retention_authors(ratio):
    return sum(1 for r in ratio if r == 0)


def num_partial_retention_authors(ratio):
    """Authors who lost some but not all of their papers (0 < rho_a < 1)."""
    return sum(1 for r in ratio if 0 < r < 1)


def lost_welfare(ratio):
    """Lambda(R) = W(P) - W(S) for the rejected set R = P \\ S; W(P) = |A| (every ratio is 1)."""
    return len(ratio) - sum(ratio)


def opt_lp(edges, n_nodes, c):
    """LP_ICML, the optimum of the natural LP relaxation (an upper bound on W(S*))."""
    value, _ = solve_lp(edges, n_nodes, c)
    return value


def certificate_u(edges, n_nodes, c, retained):
    """Certificate U of Theorem 4 for a Welfare-Greedy output S_G (retained).

    y_a = min{w_q : q in S_G ∩ P(a)} for over-cap authors saturated at the cap c (else 0),
    U = W(P0) + sum_{q in S_G \\ P0} max{w_q, sum_{a in A_c(q)} y_a} >= LP_ICML.
    Only valid when retained is the output of Welfare-Greedy.
    """
    deg = degrees(edges, n_nodes)
    over = over_cap_authors(deg, c)
    w = paper_weights(edges, deg)
    p0, _ = split_p0(edges, over)

    count = [0] * n_nodes
    y = [float("inf")] * n_nodes
    for q in retained:
        for a in edges[q]:
            if over[a]:
                count[a] += 1
                y[a] = min(y[a], w[q])
    y = [y[a] if over[a] and count[a] == c else 0.0 for a in range(n_nodes)]

    p0 = set(p0)
    u = sum(w[q] for q in p0)
    for q in retained:
        if q not in p0:
            u += max(w[q], sum(y[a] for a in edges[q] if over[a]))
    return u


def retention_quantile(ratio, q):
    """q-quantile of the per-author retention ratios (e.g. q = 0.1)."""
    return float(np.quantile(ratio, q))


def num_collateral_authors(edges, n_nodes, c, ratio):
    """Authors within the cap (d_a <= c) who still lost a paper through a co-author."""
    deg = degrees(edges, n_nodes)
    return sum(1 for a in range(n_nodes) if deg[a] <= c and ratio[a] < 1)


def loss_share_over_cap(edges, n_nodes, c, ratio):
    """Share of the lost welfare borne by over-cap authors (0 if nothing is lost)."""
    over = over_cap_authors(degrees(edges, n_nodes), c)
    total = sum(1 - r for r in ratio)
    if total == 0:
        return 0.0
    return sum(1 - ratio[a] for a in range(n_nodes) if over[a]) / total


def retention_quantile_over_cap(edges, n_nodes, c, ratio, q):
    """q-quantile of the retention ratios of over-cap authors (d_a > c) only.

    Over the whole author set the lower quantiles are almost always 1, since most authors
    lose nothing; over-cap authors are the ones every feasible policy must cut.
    """
    over = over_cap_authors(degrees(edges, n_nodes), c)
    values = [ratio[a] for a in range(n_nodes) if over[a]]
    return float(np.quantile(values, q)) if values else 1.0


AUTHOR_GROUPS = ("single", "under", "over")  # d_a = 1, 2 <= d_a <= c, d_a > c


def author_group_losses(edges, n_nodes, c, ratio):
    """Losses per author group (EQ3, Fig. 3b): single-paper authors (d_a = 1), other within-cap
    authors (2 <= d_a <= c) and over-cap authors (d_a > c).

    Returns {f"zero_{g}": authors of group g with rho_a = 0,
             f"loss_share_{g}": share of the lost welfare borne by group g (0 if none is lost)}.
    """
    deg = degrees(edges, n_nodes)
    group = ["single" if d == 1 else ("under" if d <= c else "over") for d in deg]
    zero = dict.fromkeys(AUTHOR_GROUPS, 0)
    loss = dict.fromkeys(AUTHOR_GROUPS, 0.0)
    for a in range(n_nodes):
        loss[group[a]] += 1 - ratio[a]
        if ratio[a] == 0:
            zero[group[a]] += 1
    total = sum(loss.values())
    out = {f"zero_{g}": zero[g] for g in AUTHOR_GROUPS}
    out.update({f"loss_share_{g}": loss[g] / total if total else 0.0 for g in AUTHOR_GROUPS})
    return out
