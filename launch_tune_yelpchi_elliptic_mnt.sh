#!/usr/bin/env bash
# YelpChi + Elliptic 简易调参：seed=42，每数据集 60 组；GPU0 / GPU1 双卡并行；输出 /mnt/yehang
set -euo pipefail
FMGAD_ROOT="/home/yehang/0330/FMGAD"
PY="/home/yehang/miniconda3/envs/fmgad/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="/mnt/yehang/fmgad_tune_yelpchi_elliptic_${STAMP}"
mkdir -p "$OUT"
cd "$FMGAD_ROOT"
nohup "$PY" run_tune_refined.py \
  --datasets yelpchi elliptic \
  --gpus 0 1 \
  --seeds 42 \
  --max-configs 60 \
  --sampler-seed 20260330 \
  --num-trial 1 \
  --max-workers 2 \
  --timeout-sec 28800 \
  --search-mode refined \
  --output-dir "$OUT" \
  > "$OUT/nohup_tune.log" 2>&1 &
echo $! > "$OUT/pid.txt"
echo "PID $(cat "$OUT/pid.txt")"
echo "OUT $OUT"
echo "TASKS=$((2 * 60 * 1))"
