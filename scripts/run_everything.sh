#!/bin/bash
set -e

echo "=========================================================="
echo " EdgeGuard Pipeline: Full Automation Script"
echo "=========================================================="

echo "[1/4] Establishing Environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
echo "Environment setup complete!"

echo ""
echo "[2/4] Downloading Datasets..."
# This downloads HarmBench and XSTest via the python script
python scripts/download_datasets.py

echo ""
echo "[3/4] Running Inference..."
MODELS=("q8" "q6" "q5" "q4" "q3")
DATASETS=("harmbench" "xstest")

# Optional: subset size for testing, e.g. SUBSET="--subset 10"
SUBSET=""

for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
        echo "--> Running model $m on dataset $d"
        python scripts/run_model.py --model "$m" --dataset "$d" $SUBSET
    done
done

echo ""
echo "[4/4] Generating Publication Assets..."
python evaluation/evaluate.py

echo "=========================================================="
echo " Pipeline Complete."
echo " Output Assets:"
echo " - summary: results/summary_metrics.csv"
echo " - charts:  paper/figures/"
echo " - tables:  paper/tables/"
echo "=========================================================="
