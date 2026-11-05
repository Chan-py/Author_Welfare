"""Helper functions shared by the algorithms in algorithms.py."""
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix


def degrees(edges, n_nodes):
    """d_a: number of papers of every author a."""
    deg = [0] * n_nodes
    for authors in edges:
        for a in authors:
            deg[a] += 1
    return deg


def over_cap_authors(deg, t):
    """A_t as flags: over[a] is True iff d_a > t."""
    return [d > t for d in deg]


def paper_weights(edges, deg):
    """w_p = sum_{a in A(p)} 1/d_a for every paper p (Lemma 1)."""
    return [sum(1.0 / deg[a] for a in authors) for authors in edges]


def split_p0(edges, over):
    """Split papers into P0 (no over-cap author) and the rest P \\ P0, both in id order."""
    p0, cand = [], []
    for p, authors in enumerate(edges):
        if any(over[a] for a in authors):
            cand.append(p)
        else:
            p0.append(p)
    return p0, cand


def papers_of_over_cap(edges, n_nodes, over):
    """papers_of[a] = papers of a for every over-cap author a (empty for the others)."""
    papers_of = [[] for _ in range(n_nodes)]
    for p, authors in enumerate(edges):
        for a in authors:
            if over[a]:
                papers_of[a].append(p)
    return papers_of


def top_t_trimming(cand, edges, over, w, t):
    """Top-t trimming (Lemma 3).

    Papers of cand (all outside P0) with the same over-cap author set A_t(p) are parallel;
    keep only the t heaviest of each such class (ties by paper id). Neither the optimum nor
    the Welfare-Greedy output changes.

    Returns the kept papers of cand, in their original order.
    """
    classes = {}
    for p in cand:
        key = frozenset(a for a in edges[p] if over[a])
        classes.setdefault(key, []).append(p)

    kept = set()
    for members in classes.values():
        members.sort(key=lambda p: (-w[p], p))
        kept.update(members[:t])
    return [p for p in cand if p in kept]


def build_lp(edges, n_nodes, t, trim=False):
    """Natural LP of AWMSS reduced to the over-cap hypergraph H_t (Lemma 2).

    max sum_{p in cand} w_p x_p  s.t.  sum_{p in P(a) cand} x_p <= t (a in A_t),  0 <= x <= 1

    Papers in P0 are fixed to 1 and contribute the constant W(P0). With trim=True, top-t
    trimming (Lemma 3) removes dominated parallel papers first; the optimum is unchanged.

    Returns a dict with cand (variable -> paper id), c (= w_p of cand), A (|A_t| x |cand|
    sparse matrix), p0 and base = W(P0).
    """
    deg = degrees(edges, n_nodes)
    over = over_cap_authors(deg, t)
    w = paper_weights(edges, deg)
    p0, cand = split_p0(edges, over)
    if trim:
        cand = top_t_trimming(cand, edges, over, w, t)

    row = {}  # over-cap author -> constraint index
    rows, cols = [], []
    for j, p in enumerate(cand):
        for a in edges[p]:
            if over[a]:
                rows.append(row.setdefault(a, len(row)))
                cols.append(j)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(row), len(cand)))
    return {"cand": cand, "c": np.array([w[p] for p in cand]), "A": A, "p0": p0,
            "base": sum(w[p] for p in p0)}


def solve_lp(edges, n_nodes, t, trim=False):
    """Solve the natural LP relaxation with HiGHS dual simplex (an optimal extreme point).

    Returns (OPT_LP, x) where x[p] is the LP value of every paper (1 for P0, 0 for papers
    removed by trimming).
    """
    model = build_lp(edges, n_nodes, t, trim)
    x = [0.0] * len(edges)
    for p in model["p0"]:
        x[p] = 1.0
    if not model["cand"]:
        return model["base"], x

    res = linprog(-model["c"], A_ub=model["A"], b_ub=np.full(model["A"].shape[0], t),
                  bounds=(0, 1), method="highs-ds")
    if res.status != 0:
        raise RuntimeError(f"LP not solved: {res.message}")
    for j, p in enumerate(model["cand"]):
        x[p] = float(res.x[j])
    return model["base"] - res.fun, x


def _solve_01(c, constraints, time_limit=None):
    """max c.x over x in {0,1}^n subject to constraints, with HiGHS at zero optimality gap.

    Raises if the solver stops before proving optimality (e.g. at time_limit seconds).
    """
    options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    res = milp(-c, constraints=constraints, integrality=np.ones(len(c)), bounds=Bounds(0, 1),
               options=options)
    if res.status != 0:
        raise RuntimeError(f"ILP not solved to optimality: {res.message}")
    return -res.fun, res.x > 0.5


def solve_ilp(edges, n_nodes, t, trim=True, time_limit=None):
    """Solve AWMSS exactly as a 0/1 program with the HiGHS MIP solver (zero optimality gap).

    Same model as solve_lp with x integral. Trimming (Lemma 3) keeps an optimum, so it is
    on by default to shrink the model.

    Returns (OPT, retained) with retained the sorted ids of the selected papers.
    """
    model = build_lp(edges, n_nodes, t, trim)
    if not model["cand"]:
        return model["base"], sorted(model["p0"])

    value, x = _solve_01(model["c"], LinearConstraint(model["A"], -np.inf, t), time_limit)
    chosen = [p for j, p in enumerate(model["cand"]) if x[j]]
    return model["base"] + value, sorted(model["p0"] + chosen)


def solve_cardinality_ilp(edges, n_nodes, t, trim=True, time_limit=None, worst=False):
    """Maximise the number of retained papers exactly, breaking ties by welfare.

    Lexicographic two-stage 0/1 program on the same constraints as solve_ilp:
    stage 1 finds the maximum number K of retained papers outside P0, stage 2 maximises
    welfare among selections of K such papers. Trimming keeps such an optimum (swapping a
    parallel paper for a heavier one keeps the count and does not lower welfare).
    With worst=True, stage 2 minimises welfare instead (the least favourable tie-break);
    trimming is then skipped, since it could drop the lighter papers that tie-break picks.

    Returns (number of retained papers, retained) with retained sorted.
    """
    model = build_lp(edges, n_nodes, t, trim and not worst)
    if not model["cand"]:
        return len(model["p0"]), sorted(model["p0"])

    n = len(model["cand"])
    caps = LinearConstraint(model["A"], -np.inf, t)
    k, _ = _solve_01(np.ones(n), caps, time_limit)
    count = LinearConstraint(np.ones((1, n)), round(k), np.inf)
    _, x = _solve_01(-model["c"] if worst else model["c"], [caps, count], time_limit)
    chosen = [p for j, p in enumerate(model["cand"]) if x[j]]
    return len(model["p0"]) + len(chosen), sorted(model["p0"] + chosen)


def packing_order(F, oc, x):
    """Edge order of Parekh-Pritchard Algorithm 3 (Chan-Lau ordering).

    F: fractional support (0 < x_p < 1) of an extreme point; oc[p]: over-cap authors of p.
    Repeatedly take the author with the minimum nonzero degree among the remaining papers
    (ties by author id) and remove its paper with the largest x_p (ties by paper id).
    Returns papers in removal order; they are packed in reverse.
    """
    remaining = set(F)
    degree = {}
    for p in F:
        for a in oc[p]:
            degree[a] = degree.get(a, 0) + 1
    order = []
    while remaining:
        v = min((a for a, d in degree.items() if d > 0), key=lambda a: (degree[a], a))
        e = min((p for p in remaining if v in oc[p]), key=lambda p: (-x[p], p))
        order.append(e)
        remaining.discard(e)
        for a in oc[e]:
            degree[a] -= 1
    return order


def _is_int(z, tol=1e-7):
    return abs(z - round(z)) <= tol


def _ceil(z, tol=1e-7):
    return math.ceil(z - tol)


def _take(terms, idx, amount, tol=1e-12):
    """Indices of terms (in the given order) whose mass sums to `amount`, splitting the last.

    A term is [papers, deg, lam]; splitting appends a copy holding the split-off mass.
    """
    chosen, need = [], amount
    for i in idx:
        if need <= tol:
            break
        papers, deg, lam = terms[i]
        if lam <= need + tol:
            chosen.append(i)
            need -= lam
        else:
            terms[i][2] = lam - need
            terms.append([papers[:], dict(deg), need])
            chosen.append(len(terms) - 1)
            need = 0.0
    return chosen, need


def modified_packing_step(terms, e, oc, x, load):
    """Pack x_e of paper e into the convex combination (Parekh-Pritchard, Lemma 5/6).

    terms: list of [papers, deg, lam] (integral solutions with masses); load[a] = sum of
    x over already packed papers containing a. Keeps every solution within ceil(load) at
    each author, with mass at the ceiling at most the fractional part of load.
    """
    blocked = set()
    for v in oc[e]:
        d_old = load.get(v, 0.0)
        d_new = d_old + x[e]
        if d_old <= 1e-12:
            continue                                   # case (I)
        c_old, c_new = _ceil(d_old), _ceil(d_new)
        if c_new == c_old:                             # case (II)
            blocked.update(i for i, (_, deg, _) in enumerate(terms) if deg.get(v, 0) == c_new)
        elif not (_is_int(d_new) or _is_int(d_old)):   # case (III), both non-integral
            at_ceil = [i for i, (_, deg, _) in enumerate(terms) if deg.get(v, 0) == c_old]
            # Q_v: mass 1 - x_e of at_ceil; terms already blocked count for free
            need = 1.0 - x[e] - sum(terms[i][2] for i in at_ceil if i in blocked)
            if need > 1e-12:
                chosen, _ = _take(terms, [i for i in at_ceil if i not in blocked], need)
                blocked.update(chosen)

    free = [i for i in range(len(terms)) if i not in blocked]
    chosen, short = _take(terms, free, x[e])
    if short > 1e-7:
        raise RuntimeError(f"packing step failed for paper {e}: missing mass {short}")
    for i in chosen:
        terms[i][0].append(e)
        for v in oc[e]:
            terms[i][1][v] = terms[i][1].get(v, 0) + 1
    for v in oc[e]:
        load[v] = load.get(v, 0.0) + x[e]


def over_cap_ranks(edges, deg, t):
    """|A_t(p)| for every paper p: its number of over-cap authors (d_a > t)."""
    return [sum(1 for a in authors if deg[a] > t) for authors in edges]


def fractional_components(edges, over, x, t, tol=1e-7):
    """Structure of the fractional support F = {p : tol < x_p < 1 - tol} of an LP solution (EQ2).

    Papers of F are joined through shared over-cap authors whose cap constraint is tight
    (load = t). For every connected component, the tight authors of each paper are its
    "tight rank"; if all tight ranks are <= 2 the component is a graph on authors (papers =
    edges, rank-1 papers = pendant half-edges) and its shortest odd cycle is reported.

    The paper view works for every component: papers are vertices and a tight author with
    exactly two fractional papers is an edge between them. With all values 1/2 (the case
    seen on real data) a bipartite component of this graph could be shifted (+1/2 on one
    side, -1/2 on the other), so at an extreme point every component has an odd cycle of
    papers; paper_odd_cycle is its shortest length (0 if none), and
    num_authors_over_2 counts tight authors with more than two fractional papers.

    Returns a list of dicts (largest component first) with papers, authors, max_tight_rank,
    values (x of the papers), kind ("graph" or "hyper"), odd_cycle (author-graph odd cycle
    length, 0 if bipartite, None for "hyper"), paper_odd_cycle and num_authors_over_2.
    """
    load = {}
    for p, authors in enumerate(edges):
        if x[p] > tol:
            for a in authors:
                if over[a]:
                    load[a] = load.get(a, 0.0) + x[p]
    tight = {a for a, v in load.items() if v >= t - 1e-6}
    F = [p for p in range(len(edges)) if tol < x[p] < 1 - tol]
    tight_of = {p: [a for a in edges[p] if a in tight] for p in F}

    parent = {}  # union-find over papers of F and their tight authors

    def find(v):
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for p in F:
        for a in tight_of[p]:
            parent[find(("p", p))] = find(("a", a))
        find(("p", p))
    groups = {}
    for p in F:
        groups.setdefault(find(("p", p)), []).append(p)

    comps = []
    for papers in groups.values():
        authors = sorted({a for p in papers for a in tight_of[p]})
        max_rank = max(len(tight_of[p]) for p in papers)
        comp = {"papers": sorted(papers), "authors": authors, "max_tight_rank": max_rank,
                "values": [x[p] for p in sorted(papers)]}
        if max_rank <= 2:
            comp["kind"] = "graph"
            comp["odd_cycle"] = _shortest_odd_cycle([tight_of[p] for p in papers if len(tight_of[p]) == 2])
        else:
            comp["kind"] = "hyper"
            comp["odd_cycle"] = None
        frac_of = {a: [p for p in papers if a in tight_of[p]] for a in authors}
        comp["paper_odd_cycle"] = _shortest_odd_cycle([ps for ps in frac_of.values() if len(ps) == 2])
        comp["num_authors_over_2"] = sum(len(ps) > 2 for ps in frac_of.values())
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c["papers"]), c["papers"][0]))
    return comps


def _shortest_odd_cycle(pairs):
    """Length of the shortest odd cycle of the multigraph with edges `pairs`; 0 if bipartite."""
    adj = {}
    for u, v in pairs:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    best = 0
    for root in adj:
        dist = {root: 0}
        queue = [root]
        for u in queue:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                elif dist[v] == dist[u]:  # odd closed walk of length 2 dist + 1 through root
                    length = 2 * dist[u] + 1
                    best = length if best == 0 else min(best, length)
    return best
