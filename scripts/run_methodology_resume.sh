#!/usr/bin/env bash
set -euo pipefail

cd /home/murilo/Documentos/LiDARAgroSat

.venv/bin/python scripts/run_methodology_matrix.py \
  --config configs/neon_tree_evaluation_practical.yaml \
  --device cuda \
  --stage1 fixed_watershed adaptive_watershed \
  --stage2 raster point hybrid \
  --labels field_center

.venv/bin/python scripts/analyze_methodology_suite.py \
  --config configs/neon_tree_evaluation_practical.yaml \
  --device cuda
