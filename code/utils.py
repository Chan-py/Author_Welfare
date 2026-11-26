import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "datasets")
CACHE_DIR = os.path.join(ROOT, "cache")


def load_hypergraph(dataset, data_dir=DATA_DIR):
    """Read datasets/{dataset}/network.hyp (one paper per non-empty line, author ids separated
    by commas or whitespace; an author listed twice on a paper is counted once).

    Returns:
        edges    : edges[p] = list of authors of paper p (author ids remapped to 0..|A|-1)
        node_ids : node_ids[a] = original id of author a
    """
    path = os.path.join(data_dir, dataset, "network.hyp")
    edges = []
    node_ids = []
    new_id = {}  # original id -> new id
    with open(path) as f:
        for line in f:
            toks = line.replace(",", " ").split()
            if not toks:
                continue
            edge = []
            for tok in toks:
                v = int(tok)
                if v not in new_id:
                    new_id[v] = len(node_ids)
                    node_ids.append(v)
                if new_id[v] not in edge:
                    edge.append(new_id[v])
            edges.append(edge)
    return edges, node_ids


# Memory measurement depends on the OS:
#   Linux  : /proc (exact; the peak is reset before the algorithm runs)
#   macOS  : resource.getrusage (peak since process start; no reset, so approximate)
#   Windows: psutil peak working set (peak since process start; no reset, so approximate)
if sys.platform.startswith("linux"):
    def _proc_status_bytes(key):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) * 1024  # reported in kB

    def reset_peak_memory():
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")

    def current_memory():
        return _proc_status_bytes("VmRSS:")

    def peak_memory():
        return _proc_status_bytes("VmHWM:")

else:
    if sys.platform == "win32":
        import psutil

        def peak_memory():
            return psutil.Process().memory_info().peak_wset
    else:  # macOS
        import resource

        def peak_memory():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

    def reset_peak_memory():
        pass  # not supported; the peak is measured from process start

    current_memory = peak_memory  # baseline = peak so far (approximation)


def cache_path(dataset, c, method, seed=None, cache_dir=CACHE_DIR):
    """Path without extension: {path}.hyp holds the hyperedges, {path}.json the rest.

    seed is None for deterministic methods; otherwise it is added as _s{seed}.
    """
    name = f"{dataset}_c{c}_{method}" + ("" if seed is None else f"_s{seed}")
    return os.path.join(cache_dir, name)


def save_result(path, retained, edges, node_ids, runtime, memory):
    """Save the retained hyperedge set (.hyp, original node ids) and run info (.json)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".hyp", "w") as f:
        for p in retained:
            f.write(",".join(str(node_ids[a]) for a in edges[p]) + "\n")
    with open(path + ".json", "w") as f:
        json.dump({"retained": retained, "runtime": runtime, "memory": memory}, f)


def load_result(path):
    with open(path + ".json") as f:
        return json.load(f)
