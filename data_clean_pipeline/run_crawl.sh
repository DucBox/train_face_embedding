#!/usr/bin/env bash
#
# Run ONLY the per-source clean flow for CRAWL (embed -> normalize -> dbscan ->
# template -> merge) and print a detailed stats report. Smaller scope than
# run_pipeline.sh — for iterating/debugging the crawl branch on its own.
#
#   TEST_PIPELINE=1 ./run_crawl.sh      # local logic test on synthetic fixtures
#   TEST_PIPELINE=0 NPROC=8 ./run_crawl.sh
#
set -euo pipefail
cd "$(dirname "$0")"
export TEST_PIPELINE="${TEST_PIPELINE:-1}"
NPROC="${NPROC:-8}"
PY="${PY:-python3}"
SRC=crawl

echo "=== crawl-only clean | TEST_PIPELINE=$TEST_PIPELINE ==="

# Stage 01 — embed crawl
if [ "$TEST_PIPELINE" = "1" ]; then
  echo "--- [test] synthetic fixtures (stands in for embed) ---"
  $PY -c "from config import CFG; from common import make_fixtures; make_fixtures(CFG)"
else
  echo "--- embed crawl from S3 (multi-GPU, per-id checkpoint) ---"
  torchrun --nproc_per_node="$NPROC" embed_s3.py
fi

# Stages 02-05
$PY normalize.py --src "$SRC"; $PY verify.py --stage normalize --src "$SRC"
$PY dbscan.py    --src "$SRC"; $PY verify.py --stage dbscan    --src "$SRC"
$PY template.py  --src "$SRC"; $PY verify.py --stage template  --src "$SRC"
$PY merge.py     --src "$SRC"; $PY verify.py --stage merge     --src "$SRC"

# Consolidated stats report
$PY stats.py --src "$SRC"
