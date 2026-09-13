#!/usr/bin/env bash
# Runs the full EMSA training + evaluation pipeline on all three datasets.
# Edit configs/config.yaml first to point at your local dataset paths.
set -euo pipefail

CONFIG="${1:-configs/config.yaml}"

echo "=== EMSA full pipeline (config: ${CONFIG}) ==="

python -m evaluation.run_all --config "${CONFIG}" --datasets vogue pixelrec mmshop

echo "=== Done. See results/ for all tables and results/figures/ for plots. ==="
