#!/usr/bin/env bash
# Hard-case mining pipeline (3 stages):
#   Stage 1: embed_dataset.py  — embed all images -> Parquet (multi-GPU, fp32)
#   Stage 2: find_hard_thresholds.py — find cosine threshold at FMR, output false_accept/reject CSV
#   Stage 3: generate_hard_cases.py  — produce hard_class_neighbors + hard_images_review CSV
#
# Usage:
#   ./run_hard_mining.sh
# All tunables are in the CONFIG section below. Override via env:
#   NPROC=8 FMR=1e-6 ./run_hard_mining.sh

set -euo pipefail

# ============================================================
# CONFIG — edit these before running
# ============================================================
CONFIG="${CONFIG:-configs/wf42m_pfc03_40epoch_64gpu_vit_l}"
MODEL_WEIGHT="${MODEL_WEIGHT:-/workspace/data/workspace/face_embedding/output/model.pt}"

# Directory containing the final .rec/.idx files to embed.
# Point to rec_out/ from data_clean_pipeline (auto-detected by train_synthetic_clean.rec),
# or to the classic training data directory.
REC_DIR="${REC_DIR:-/workspace/data/workspace/datasets/clean_loop/model_viettelai005/rec_out}"

# Where all outputs go (Parquet embeddings + CSV artifacts).
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/data/workspace/datasets/clean_loop/model_viettelai005/hard_mining}"

# Stage 1 tuning
NPROC="${NPROC:-4}"            # GPUs for torchrun
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FLUSH_ROWS="${FLUSH_ROWS:-500000}"

# Stage 2 tuning
FMR="${FMR:-1e-6}"
TOPK="${TOPK:-50}"             # nearest-medoid candidates per class for impostor search
DEVICE="${DEVICE:-cuda}"

# ============================================================
# HELPERS
# ============================================================
log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

check_file() { [[ -f "$1" ]] || die "Required file not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "Required directory not found: $1"; }

# ============================================================
# PRE-FLIGHT
# ============================================================
check_file "$MODEL_WEIGHT"
check_dir  "$REC_DIR"
mkdir -p "$OUTPUT_DIR"

log "Hard-case mining pipeline starting"
log "  config      : $CONFIG"
log "  model       : $MODEL_WEIGHT"
log "  rec_dir     : $REC_DIR"
log "  output_dir  : $OUTPUT_DIR"
log "  GPUs        : $NPROC"
log "  FMR target  : $FMR"

# ============================================================
# STAGE 1 — embed
# ============================================================
EMBED_DONE="$OUTPUT_DIR/.embed_done"
if [[ -f "$EMBED_DONE" ]]; then
    log "Stage 1 [embed] already done (found $EMBED_DONE), skipping"
else
    log "Stage 1/3 [embed] starting — torchrun with $NPROC GPU(s)..."
    torchrun \
        --nproc_per_node="$NPROC" \
        --master_port=29501 \
        embed_dataset.py "$CONFIG" \
        --weight      "$MODEL_WEIGHT" \
        --rec-dir     "$REC_DIR" \
        --output-dir  "$OUTPUT_DIR" \
        --batch-size  "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --flush-rows  "$FLUSH_ROWS"
    touch "$EMBED_DONE"
    log "Stage 1/3 [embed] done"
fi

# ============================================================
# STAGE 2 — find thresholds
# ============================================================
THRESHOLD_DONE="$OUTPUT_DIR/.threshold_done"
if [[ -f "$THRESHOLD_DONE" ]]; then
    log "Stage 2 [find_hard_thresholds] already done, skipping"
else
    log "Stage 2/3 [find_hard_thresholds] FMR=$FMR, topk=$TOPK, device=$DEVICE..."
    python3 find_hard_thresholds.py "$CONFIG" \
        --embeddings-dir "$OUTPUT_DIR" \
        --output-dir     "$OUTPUT_DIR" \
        --fmr            "$FMR" \
        --topk           "$TOPK" \
        --device         "$DEVICE"
    touch "$THRESHOLD_DONE"
    log "Stage 2/3 [find_hard_thresholds] done"
fi

# ============================================================
# STAGE 3 — generate artifacts
# ============================================================
log "Stage 3/3 [generate_hard_cases] ..."
python3 generate_hard_cases.py \
    --input-dir  "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR"
log "Stage 3/3 [generate_hard_cases] done"

# ============================================================
# SUMMARY
# ============================================================
log "============ HARD MINING COMPLETE ============"
log "Outputs in $OUTPUT_DIR:"
for f in false_accept.csv false_reject.csv hard_class_neighbors.csv hard_images_review.csv; do
    if [[ -f "$OUTPUT_DIR/$f" ]]; then
        lines=$(wc -l < "$OUTPUT_DIR/$f")
        log "  $f : $((lines - 1)) rows"
    fi
done
