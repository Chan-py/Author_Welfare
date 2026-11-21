#!/bin/bash
# Run main.py then eval.py for one setting.
# usage: ./run.sh <dataset> <c> [out_csv] [method] [seed]
#   c: per-author paper cap
#   e.g. ./run.sh real/icde2024 5
#        ./run.sh real/icde2024 3 ../results/results.csv welfare_reject
#        ./run.sh real/icde2024 3 ../results/results.csv random_reject 7   (seed: random_reject only)
set -e
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
    echo "usage: $0 <dataset> <c> [out_csv] [method] [seed]"
    exit 1
fi

DATASET=$1
CAP=$2
OUT=${3:-../results/results.csv}
METHOD=${4:-welfare_greedy}
SEED=${5:-0}

python3 main.py --dataset "$DATASET" --c "$CAP" --method "$METHOD" --seed "$SEED"
python3 eval.py --dataset "$DATASET" --c "$CAP" --method "$METHOD" --seed "$SEED" --out "$OUT"
