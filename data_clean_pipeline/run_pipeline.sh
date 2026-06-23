#!/usr/bin/env bash
#
# End-to-end orchestration of the dataset self-cleaning loop.
#
#   TEST_PIPELINE=1 ./run_pipeline.sh      # local logic test on synthetic fixtures
#   TEST_PIPELINE=0 ./run_pipeline.sh      # real run (needs GPUs / S3 / mxnet / faiss)
#
# All knobs live in config.py. This script only wires stages + verify gates and
# stops on the first failure (set -e). Every stage is resumable, so re-running
# after a crash skips finished work.
set -euo pipefail
cd "$(dirname "$0")"

export TEST_PIPELINE="${TEST_PIPELINE:-1}"
NPROC="${NPROC:-8}"                 # GPUs for the real embed steps
PY="${PY:-python3}"

echo "=== clean-loop pipeline | TEST_PIPELINE=$TEST_PIPELINE ==="
$PY -c "from config import CFG; print('work_dir =', CFG.work_dir)"

# --------------------------------------------------------------------------- #
# Stage 01 — EMBED (per source, with the new model)
# --------------------------------------------------------------------------- #
if [ "$TEST_PIPELINE" = "1" ]; then
  echo "--- [test] generating synthetic fixtures (stands in for embed) ---"
  $PY -c "from config import CFG; from common import make_fixtures; make_fixtures(CFG)"
else
  echo "--- embed webface / public (rec, multi-GPU) ---"
  torchrun --nproc_per_node="$NPROC" embed_rec.py --src webface
  torchrun --nproc_per_node="$NPROC" embed_rec.py --src public
  echo "--- embed crawl (S3, multi-GPU, per-id checkpoint) ---"
  torchrun --nproc_per_node="$NPROC" embed_s3.py
fi

# --------------------------------------------------------------------------- #
# Stages 02-05 — per-source: normalize -> dbscan -> template -> merge(internal)
# --------------------------------------------------------------------------- #
for SRC in webface public crawl; do
  echo "=== per-source clean: $SRC ==="
  $PY normalize.py --src "$SRC"; $PY verify.py --stage normalize --src "$SRC"
  $PY dbscan.py    --src "$SRC"; $PY verify.py --stage dbscan    --src "$SRC"
  $PY template.py  --src "$SRC"; $PY verify.py --stage template  --src "$SRC"
  $PY merge.py     --src "$SRC"; $PY verify.py --stage merge     --src "$SRC"
done

# --------------------------------------------------------------------------- #
# Stages 06-08 — global: template -> merge(symmetric) -> reindex
# --------------------------------------------------------------------------- #
echo "=== global merge ==="
$PY template.py --global
$PY merge.py    --global; $PY verify.py --stage merge --global
$PY reindex.py            ; $PY verify.py --stage reindex

# --------------------------------------------------------------------------- #
# Stage 09 — write the 3 .rec files (one contiguous id range)
# --------------------------------------------------------------------------- #
echo "=== write rec ==="
$PY write_rec.py

echo "=== DONE ==="
$PY -c "from common import meta_read; from config import CFG; import json; \
print(json.dumps(meta_read(CFG.path_meta()), indent=2))"
