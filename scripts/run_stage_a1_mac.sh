#!/usr/bin/env bash
set -euo pipefail

LEXICON="data/lexicon_en_glove_covered.tsv"
MAX_WORDS=29571
EPOCHS=30
BATCH_SIZE=64
SEED=0
TRAIN_NOISE=0.0
GATE_ALPHA=4.0

CKPT_ROOT="checkpoints/gridsearch_stage_a1"
OUT_ROOT="outputs/gridsearch_stage_a1"
LOG_ROOT="${OUT_ROOT}/logs"

mkdir -p "$CKPT_ROOT" "$OUT_ROOT" "$LOG_ROOT"

run_one () {
    local RUN_ID="$1"
    local LR="$2"
    local TF_RATIO="$3"

    local CKPT="${CKPT_ROOT}/${RUN_ID}.pt"
    local TRAIN_OUT="${OUT_ROOT}/${RUN_ID}/train"
    local EVAL_AR_OUT="${OUT_ROOT}/${RUN_ID}/train_val_ar"
    local WFE_AR_OUT="${OUT_ROOT}/${RUN_ID}/wfe_ar"
    local LOG="${LOG_ROOT}/${RUN_ID}.log"

    echo "============================================================"
    echo "START ${RUN_ID}"
    echo "LR=${LR}"
    echo "TF_RATIO=${TF_RATIO}"
    echo "EPOCHS=${EPOCHS}"
    echo "TRAIN_NOISE=${TRAIN_NOISE}"
    echo "GATE_ALPHA=${GATE_ALPHA}"
    echo "START_TIME=$(date)"
    echo "============================================================"

    mkdir -p "$TRAIN_OUT" "$EVAL_AR_OUT" "$WFE_AR_OUT"

    {
        echo "============================================================"
        echo "TRAINING ${RUN_ID}"
        echo "START_TIME=$(date)"
        echo "============================================================"

        time python scripts/train_checkpoint.py \
            --lexicon_path "$LEXICON" \
            --max_words "$MAX_WORDS" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --seed "$SEED" \
            --lr "$LR" \
            --teacher_forcing_ratio "$TF_RATIO" \
            --interference_noise "$TRAIN_NOISE" \
            --gate_alpha "$GATE_ALPHA" \
            --ckpt "$CKPT" \
            --out_dir "$TRAIN_OUT"

        echo ""
        echo "============================================================"
        echo "TRAIN/VAL AUTOREGRESSIVE EVAL ${RUN_ID}"
        echo "START_TIME=$(date)"
        echo "============================================================"

        python scripts/evaluate_train_lexicon_ceiling.py \
            --ckpt "$CKPT" \
            --lexicon_path "$LEXICON" \
            --out_dir "$EVAL_AR_OUT" \
            --include_val \
            --decode autoregressive

        echo ""
        echo "============================================================"
        echo "WFE AUTOREGRESSIVE NO-NOISE EVAL ${RUN_ID}"
        echo "START_TIME=$(date)"
        echo "============================================================"

        python scripts/external_eval.py \
            --ckpt "$CKPT" \
            --out_dir "$WFE_AR_OUT" \
            --decode autoregressive \
            --wfe_only

        echo ""
        echo "============================================================"
        echo "DONE ${RUN_ID}"
        echo "END_TIME=$(date)"
        echo "============================================================"
    } 2>&1 | tee "$LOG"
}

run_one "A1_lr1e-3_tf1p0"  "0.001"  "1.0"
run_one "A2_lr1e-3_tf0p2"  "0.001"  "0.2"
run_one "A3_lr1e-3_tf0p0"  "0.001"  "0.0"
run_one "A4_lr5e-4_tf1p0"  "0.0005" "1.0"
run_one "A5_lr5e-4_tf0p2"  "0.0005" "0.2"
run_one "A6_lr5e-4_tf0p0"  "0.0005" "0.0"

echo ""
echo "============================================================"
echo "STAGE A1 COMPLETE"
echo "END_TIME=$(date)"
echo "============================================================"
