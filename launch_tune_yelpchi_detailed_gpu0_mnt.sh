#!/usr/bin/env bash
# YelpChi 精细调参（detailed 网格）：单卡 GPU0，seed=42，nohup 后台；输出 /mnt/yehang
set -euo pipefail
FMGAD_ROOT="/home/yehang/0330/FMGAD"
PY="/home/yehang/miniconda3/envs/fmgad/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="/mnt/yehang/fmgad_tune_yelpchi_detailed_gpu0_${STAMP}"
mkdir -p "$OUT"
cd "$FMGAD_ROOT"
nohup "$PY" run_tune_refined.py \
  --datasets yelpchi \
  --gpus 0 \
  --seeds 42 \
  --max-configs 100 \
  --sampler-seed 20260331 \
  --num-trial 1 \
  --max-workers 1 \
  --timeout-sec 28800 \
  --search-mode detailed \
  --output-dir "$OUT" \
  > "$OUT/nohup_tune.log" 2>&1 &
echo $! > "$OUT/pid.txt"
echo "PID $(cat "$OUT/pid.txt")"
echo "OUT $OUT"
echo "TASKS=100 (yelpchi x 100 x seed42)"
