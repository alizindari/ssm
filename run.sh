#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Phase Transitions in Gated State Space Models — Parity Task
# ═══════════════════════════════════════════════════════════════
# Edit the variables below to configure the experiment.
# Then run: bash run.sh

# ── Model ──
MODEL="gru"  # "gru" or "mamba"

# ── Main experiment: vary hidden dimension ──
MAIN_HIDDEN_DIMS="1,2,3,4,6,8,12,16"
MAIN_SEQ_LEN=8
MAIN_SEEDS=1

# ── Secondary experiment: vary sequence length ──
SEC_SEQ_LENS="4,6,8,10,12,16"
SEC_HIDDEN_DIM=4
SEC_SEEDS=1

# ── Training ──
NUM_EPOCHS=3000
BATCH_SIZE=64
LR=0.001

# ── Testing ──
TEST_LEN=16
TEST_SIZE=500
LARGE_N_THRESHOLD=12
LARGE_N_SAMPLE=4096

# ── Logging & output ──
LOG_INTERVAL=10
DPI=200
DEVICE="cpu"
FIG_DIR="../figures"
RESULTS_FILE="../ssm_grokking_results.pkl"

# ═══════════════════════════════════════════════════════════════

cd "$(dirname "$0")/src"

python3 main.py \
    --model "$MODEL" \
    --main-hidden-dims "$MAIN_HIDDEN_DIMS" \
    --main-seq-len "$MAIN_SEQ_LEN" \
    --main-seeds "$MAIN_SEEDS" \
    --sec-seq-lens "$SEC_SEQ_LENS" \
    --sec-hidden-dim "$SEC_HIDDEN_DIM" \
    --sec-seeds "$SEC_SEEDS" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --test-len "$TEST_LEN" \
    --test-size "$TEST_SIZE" \
    --large-n-threshold "$LARGE_N_THRESHOLD" \
    --large-n-sample "$LARGE_N_SAMPLE" \
    --log-interval "$LOG_INTERVAL" \
    --dpi "$DPI" \
    --device "$DEVICE" \
    --fig-dir "$FIG_DIR" \
    --results-file "$RESULTS_FILE"
