import heapq
import random

from function import (degrees, modified_packing_step, over_cap_authors, packing_order, paper_weights,
                      papers_of_over_cap, solve_cardinality_ilp, solve_ilp, solve_lp, split_p0,
                      top_c_trimming)


def welfare_greedy(edges, n_nodes, c, trim=False):
    """Welfare-Greedy, WGA (Algorithm 1, Section 4.1).

    Examine papers in non-increasing order of w_p = sum_{a in A(p)} 1/d_a and retain a
    paper if no over-cap author (d_a > c) of it has reached the cap c. Ties are broken by
    paper id.

    Two output-preserving refinements from the running-time paragraph of Section 4.1:
    papers in P0 (no over-cap author) are retained before sorting, and the loop stops
    once every over-cap author is saturated (every remaining paper would be rejected).
    With trim=True, top-c trimming (Lemma 3) is applied before sorting; the output is
    the same.

    Returns sorted ids of retained papers.
    """
    deg = degrees(edges, n_nodes)   # d_a: number of papers of author a
    over = over_cap_authors(deg, c)  # A_c: over-cap authors

    # Remove papers in P0 from the candidate set.
    retained, cand = split_p0(edges, over)  # cand = P \ P0

    w = paper_weights(edges, deg)
    if trim:
        cand = top_c_trimming(cand, edges, over, w, c)
    cand.sort(key=lambda p: -w[p])  # stable: ties keep paper id order

    count = [0] * n_nodes  # c_a: retained papers of over-cap author a
    n_open = sum(over)     # over-cap authors with c_a < c
    for p in cand:
        if n_open == 0:
            break
        oc = [a for a in edges[p] if over[a]]   # A_c(p): over-cap authors of paper p
        if all(count[a] < c for a in oc):
            retained.append(p)
            for a in oc:
                count[a] += 1
                if count[a] == c:
                    n_open -= 1

    return sorted(retained)


def welfare_reject(edges, n_nodes, c):
    """Welfare-Reject (Section 4.3).

    An over-cap author is open while its unrejected papers exceed the cap c, and c_p is the
    number of open authors of an unrejected paper p. Repeatedly reject the unrejected paper
    with c_p >= 1 that minimises w_p / c_p (ties by paper id) until no author is open.
    c_p only decreases, so a lazy priority queue suffices.

    Returns sorted ids of the retained papers S_R = P \\ R (R: rejected papers).
    """
    deg = degrees(edges, n_nodes)
    over = over_cap_authors(deg, c)
    papers_of = papers_of_over_cap(edges, n_nodes, over)
    w = paper_weights(edges, deg)

    remaining = deg[:]                      # unrejected papers of each author
    n_open = [sum(over[a] for a in authors) for authors in edges]  # c_p: open authors of p
    heap = [(w[p] / n_open[p], p, n_open[p]) for p in range(len(edges)) if n_open[p] > 0]
    heapq.heapify(heap)

    rejected = [False] * len(edges)
    while heap:
        _, p, k = heapq.heappop(heap)
        if rejected[p] or k != n_open[p]:
            continue  # stale entry
        rejected[p] = True
        for a in edges[p]:
            remaining[a] -= 1
            if over[a] and remaining[a] == c:  # a closes
                for q in papers_of[a]:
                    if not rejected[q]:
                        n_open[q] -= 1
                        if n_open[q] > 0:
                            heapq.heappush(heap, (w[q] / n_open[q], q, n_open[q]))

    return [p for p in range(len(edges)) if not rejected[p]]


def random_reject(edges, n_nodes, c, seed=0):
    """Random desk rejection (Proposition 1), the policy used in practice.

    Over-cap authors are processed in a uniformly random order; while an author still
    has more than c unrejected papers, one of them is rejected uniformly at random.

    Returns sorted ids of retained papers.
    """
    rng = random.Random(seed)
    deg = degrees(edges, n_nodes)
    over = over_cap_authors(deg, c)
    papers_of = papers_of_over_cap(edges, n_nodes, over)

    remaining = deg[:]
    rejected = [False] * len(edges)
    authors = [a for a in range(n_nodes) if over[a]]
    rng.shuffle(authors)
    for a in authors:
        pool = papers_of[a][:]
        while remaining[a] > c:
            i = rng.randrange(len(pool))
            p = pool[i]
            pool[i] = pool[-1]
            pool.pop()
            if rejected[p]:
                continue  # already rejected through a co-author
            rejected[p] = True
            for b in edges[p]:
                remaining[b] -= 1

    return [p for p in range(len(edges)) if not rejected[p]]


def exact_ilp(edges, n_nodes, c):
    """Exact optimum S* of AWMSS by integer programming (see function.solve_ilp).

    Returns sorted ids of retained papers.
    """
    _, retained = solve_ilp(edges, n_nodes, c)
    return retained


def max_cardinality(edges, n_nodes, c):
    """Cardinality maximisation, the "maximise retained papers" policy of EQ3.

    Exact optimum by integer programming; among selections with the most papers, one of
    maximum welfare is returned (see function.solve_cardinality_ilp).

    Returns sorted ids of retained papers.
    """
    _, retained = solve_cardinality_ilp(edges, n_nodes, c)
    return retained


def max_cardinality_worst(edges, n_nodes, c):
    """max_cardinality with the least favourable tie-break: among selections with the most
    papers, one of minimum welfare (EQ3).

    Returns sorted ids of retained papers.
    """
    _, retained = solve_cardinality_ilp(edges, n_nodes, c, worst=True)
    return retained


def iterated_packing(edges, n_nodes, c, return_ratio=False):
    """LP-based iterated packing, LIP (Section 4.2, Corollary 1; Parekh & Pritchard, Algorithm 3).

    Solve the natural LP of H_c for an optimal extreme point x*. Papers with x*_p = 1 are
    retained; on the fractional support F = {p : 0 < x*_p < 1}, x* is written as a combination of
    integral b-matchings with total mass k - 1 + 1/k, where k = max_{p in F} |A_c(p)| <= k_c,
    by packing F in reverse Chan-Lau order with modified packing steps. The heaviest
    solution of the combination is retained, so its weight is at least
    (sum_{p in F} w_p x*_p) / (k - 1 + 1/k). The output is S_PP = S_IP ∪ P0, where S_IP is
    the b-matching of H_c (the papers with x*_p = 1 and the chosen papers of F).

    Returns sorted ids of retained papers (and the ratio k - 1 + 1/k if return_ratio).
    """
    deg = degrees(edges, n_nodes)
    over = over_cap_authors(deg, c)
    w = paper_weights(edges, deg)
    p0, cand = split_p0(edges, over)
    _, x = solve_lp(edges, n_nodes, c)

    oc = {p: [a for a in edges[p] if over[a]] for p in cand}  # A_c(p)
    ones = [p for p in cand if x[p] >= 1 - 1e-9]
    F = [p for p in cand if 1e-9 < x[p] < 1 - 1e-9]
    if not F:
        return (sorted(p0 + ones), 1.0) if return_ratio else sorted(p0 + ones)

    k = max(len(oc[p]) for p in F)
    ratio = k - 1 + 1 / k
    terms = [[[], {}, ratio]]  # [papers, deg, lam]: start from the empty solution
    load = {}
    for e in reversed(packing_order(F, oc, x)):
        modified_packing_step(terms, e, oc, x, load)

    best = max(terms, key=lambda term: (sum(w[p] for p in term[0]), -len(term[0])))[0]
    retained = sorted(p0 + ones + best)
    return (retained, ratio) if return_ratio else retained


METHODS = {
    "welfare_greedy": welfare_greedy,
    "welfare_greedy_trim": lambda edges, n_nodes, c: welfare_greedy(edges, n_nodes, c, trim=True),
    "welfare_reject": welfare_reject,
    "random_reject": random_reject,
    "exact_ilp": exact_ilp,
    "iterated_packing": iterated_packing,
    "max_cardinality": max_cardinality,
    "max_cardinality_worst": max_cardinality_worst,
}

# methods that take a seed argument
SEEDED_METHODS = {"random_reject"}
