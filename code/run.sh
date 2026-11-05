#!/bin/bash
# Run main.py then eval.py for one setting.
# usage: ./run.sh <dataset> <t> [out_csv] [method] [seed]
#   e.g. ./run.sh real/icde2024 5
#        ./run.sh real/iclr2026 20 ../results/eq3.csv welfare_greedy
#        ./run.sh real/iclr2026 20 ../results/eq4.csv random_reject 3   (seed: random_reject only)
set -e
cd "$(dirname "$0")"

if [ $# -lt 2 ]; then
    echo "usage: $0 <dataset> <t> [out_csv] [method] [seed]"
    exit 1
fi

DATASET=$1
T=$2
OUT=${3:-../results/results.csv}
METHOD=${4:-welfare_greedy}
SEED=${5:-0}

python3 main.py --dataset "$DATASET" --t "$T" --method "$METHOD" --seed "$SEED"
python3 eval.py --dataset "$DATASET" --t "$T" --method "$METHOD" --seed "$SEED" --out "$OUT"
