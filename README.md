# Author Welfare

Code for the paper on author welfare under a per-author paper cap *c*.

## Requirements
- Python 3.10+, Linux (memory measurement reads `/proc`)
- `pip install -r requirements.txt`

## Structure
```
code/
  main.py        # run one algorithm, save selected papers to cache/
  eval.py        # compute measures from cache/ and append a row to a csv
  algorithms.py  # algorithms (welfare_greedy, welfare_reject, iterated_packing, exact_ilp, ...)
  function.py    # helpers: degrees, paper weights, LP/ILP, trimming, packing
  measures.py    # evaluation measures (welfare, retention ratio, ...)
  utils.py       # hypergraph loading, cache I/O
  run.sh         # main.py + eval.py for one setting
datasets/        # datasets/{dataset}/network.hyp
```

## Data format
`datasets/{dataset}/network.hyp`: one paper per line, comma-separated author ids (an author listed twice on a paper is counted once).
```
1,2,3
4,5
```

## Usage
```bash
cd code
python3 main.py --dataset real/icde2024 --c 5 --method welfare_greedy
python3 eval.py --dataset real/icde2024 --c 5 --method welfare_greedy --out ../results/results.csv
# or both at once
./run.sh real/icde2024 5 ../results/results.csv welfare_greedy
```

- `--c`: per-author paper cap *c*
- `--method`: `welfare_greedy`, `welfare_greedy_trim`, `welfare_reject`, `random_reject` (uses `--seed`), `exact_ilp`, `iterated_packing`, `max_cardinality`, `max_cardinality_worst`
- Output: `cache/{dataset}_c{c}_{method}.hyp` (retained papers) and `.json` (retained ids, runtime, memory); `eval.py` appends measures (welfare, min retention ratio, # zero-retention authors, W(S) / LP_ICML, ...) to the csv.
