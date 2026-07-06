#!/usr/bin/env bash
# PANCDR 論文結果重現流程（在 PANCDR Docker container 內執行）
#
# 用法（從 host）:
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh check
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh smoke
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh tcga
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh baseline
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh nested
#   docker exec -it PANCDR bash /workspace/PANCDR/scripts/reproduce_pancdr.sh all
#
# 設計思路:
#   1. check  — 驗證 Python/CUDA 與資料完整性，避免長時間訓練後才發現缺檔
#   2. smoke  — 只跑 1 次訓練，確認資料管線與 GPU 可正常 forward/backward
#   3. tcga   — 官方 run_PANCDR.py，100 次獨立訓練，對應論文 Table 1 TCGA PANCDR
#   4. baseline — DeepCDR 對照組
#   5. nested — GDSC 10-fold outer CV，通常最耗時，放最後

set -euo pipefail

PROJECT_ROOT="/workspace/PANCDR"
SRC_DIR="${PROJECT_ROOT}/src"
CHECKPOINT_TCGA="${PROJECT_ROOT}/checkpoint/TCGA"

step="${1:-check}"

log() { echo "[reproduce] $*"; }

check_env() {
    cd "${SRC_DIR}"
    log "working directory: $(pwd)"

    python -V
    python - <<'PY'
import torch, pandas, hickle
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
print("pandas:", pandas.__version__)
print("hickle:", hickle.__version__)
PY
}

prepare_data() {
  cd "${PROJECT_ROOT}"
  # README 提供 zip，程式讀 csv；若只有 zip 則解壓到 TCGA/
  if [ -f data/TCGA/Pretrain_TCGA_expr_702_01A.zip ] && \
     [ ! -f data/TCGA/Pretrain_TCGA_expr_702_01A.csv ]; then
    log "unzipping Pretrain_TCGA_expr_702_01A.zip ..."
    unzip -o data/TCGA/Pretrain_TCGA_expr_702_01A.zip -d data/TCGA/
  fi

  log "GDSC data:"
  ls -lh data/GDSC/*.csv data/GDSC/drug_graph_feat 2>/dev/null | head -20 || true

  log "TCGA data:"
  ls -lh data/TCGA/*.csv data/TCGA/*.txt data/TCGA/drug_graph_feat 2>/dev/null | head -20 || true

  # 訓練腳本實際依賴的關鍵檔案
  for f in \
    data/GDSC/GDSC_binary_response_151.csv \
    data/GDSC/GDSC_expr_z_702.csv \
    data/GDSC/drug_graph_feat \
    data/TCGA/Pretrain_TCGA_expr_702_01A.csv \
    data/TCGA/TCGA_expr_z_702.csv \
    data/TCGA/TCGA_response_new.csv \
    src/tuned_hyperparameters/TCGA_CV_params.csv; do
    if [ ! -e "${PROJECT_ROOT}/${f}" ]; then
      echo "ERROR: missing required file: ${f}" >&2
      exit 1
    fi
  done
  log "required data files: OK"
}

run_smoke() {
  prepare_data
  mkdir -p "${CHECKPOINT_TCGA}"
  cd "${SRC_DIR}"
  log "smoke test: 1 training iteration (see run_PANCDR_smoke.py)"
  python run_PANCDR_smoke.py
  log "smoke results:"
  ls -lh TCGA_smoke_results.csv
  cat TCGA_smoke_results.csv
}

run_tcga() {
  prepare_data
  mkdir -p "${CHECKPOINT_TCGA}"
  cd "${SRC_DIR}"
  log "TCGA 100-train PANCDR (論文 Table 1 目標 AUC ~ 0.7106)"
  python run_PANCDR.py
  log "results:"
  ls -lh TCGA_100train_results.csv
  tail -n 5 TCGA_100train_results.csv
  python - <<'PY'
import pandas as pd
df = pd.read_csv("TCGA_100train_results.csv", index_col=0)
runs = df.iloc[:-1, 0]
print("mean AUC:", runs.mean())
print("std  AUC:", runs.std())
print("論文參考: AUC 0.7106 ± 0.0246, ACC 0.6686, F1 0.6704")
PY
}

run_baseline() {
  prepare_data
  mkdir -p "${CHECKPOINT_TCGA}"
  cd "${SRC_DIR}"
  log "TCGA 100-train DeepCDR baseline"
  python run_baseline.py
  ls -lh Base_TCGA_100train_results.csv
  tail -n 5 Base_TCGA_100train_results.csv
}

run_nested() {
  prepare_data
  cd "${SRC_DIR}"
  log "GDSC 10-fold outer nested CV (論文 Table 1 目標 AUC ~ 0.7970)"
  python run_PANCDR_nested.py
  ls -lh GDSC_nested.csv
  cat GDSC_nested.csv
}

case "${step}" in
  check)
    check_env
    prepare_data
    ;;
  smoke)
    check_env
    run_smoke
    ;;
  tcga)
    check_env
    run_tcga
    ;;
  baseline)
    check_env
    run_baseline
    ;;
  nested)
    check_env
    run_nested
    ;;
  all)
    check_env
    run_smoke
    run_tcga
    run_baseline
    run_nested
    ;;
  *)
    echo "usage: $0 {check|smoke|tcga|baseline|nested|all}" >&2
    exit 1
    ;;
esac

log "done: ${step}"
