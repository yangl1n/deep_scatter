#!/bin/bash
# run_cifar2.sh — 21 experiments in 3 sequential batches of 7 (one per GPU)
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scatter_net

COMMON="--lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 50"
LOG_DIR="logs2"
mkdir -p "$LOG_DIR"

run_batch() {
    local BATCH_NAME=$1
    shift
    local -n BATCH_EXPS=$1

    echo ""
    echo "========================================"
    echo "  $BATCH_NAME"
    echo "========================================"

    local PIDS=()
    for GPU in "${!BATCH_EXPS[@]}"; do
        echo "[GPU $GPU] Starting: ${BATCH_EXPS[$GPU]}"
        CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON ${BATCH_EXPS[$GPU]} \
            > "$LOG_DIR/${BATCH_NAME}_gpu${GPU}.log" 2>&1 &
        PIDS+=($!)
    done

    echo "PIDs: ${PIDS[*]}"

    local FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[GPU $i] Done (success)"
        else
            echo "[GPU $i] FAILED (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
    done

    if [ "$FAILED" -eq 0 ]; then
        echo "$BATCH_NAME: all succeeded."
    else
        echo "$BATCH_NAME: $FAILED experiment(s) failed. Check logs."
    fi
}

# =========================================================================
# Batch 1: Fill gaps in basic grid (modulus × L × depth)
# =========================================================================
declare -A B1
B1[0]="--n-blocks 2 --modulus-type phase_relu --L 8 --save-dir cifar_runs/2b_prelu"
B1[1]="--n-blocks 2 --modulus-type complex_modulus --L 4 --save-dir cifar_runs/2b_cmod_L4"
B1[2]="--n-blocks 2 --modulus-type phase_relu --L 4 --save-dir cifar_runs/2b_prelu_L4"
B1[3]="--n-blocks 3 --modulus-type complex_modulus --L 4 --save-dir cifar_runs/3b_cmod_L4"
B1[4]="--n-blocks 3 --modulus-type phase_relu --L 4 --save-dir cifar_runs/3b_prelu_L4"
B1[5]="--n-blocks 4 --modulus-type complex_modulus --L 8 --lowpass-last --mixing-horizon 27 --kernel-size 5 --save-dir cifar_runs/4b_cmod_lp"
B1[6]="--n-blocks 4 --modulus-type complex_modulus --L 4 --lowpass-last --save-dir cifar_runs/4b_cmod_L4_lp"

run_batch "batch1" B1

# =========================================================================
# Batch 2: Kernel size, global-avg-pool, mixing-horizon effects
# =========================================================================
declare -A B2
B2[0]="--n-blocks 3 --modulus-type phase_relu --L 8 --kernel-size 5 --save-dir cifar_runs/3b_prelu_k5"
B2[1]="--n-blocks 3 --modulus-type complex_modulus --L 8 --kernel-size 5 --save-dir cifar_runs/3b_cmod_k5"
B2[2]="--n-blocks 4 --modulus-type phase_relu --L 4 --lowpass-last --save-dir cifar_runs/4b_prelu_L4_lp"
B2[3]="--n-blocks 3 --modulus-type phase_relu --L 8 --global-avg-pool --save-dir cifar_runs/3b_prelu_gap"
B2[4]="--n-blocks 4 --modulus-type phase_relu --L 8 --lowpass-last --mixing-horizon 27 --kernel-size 5 --global-avg-pool --save-dir cifar_runs/4b_prelu_lp_gap"
B2[5]="--n-blocks 3 --modulus-type phase_relu --L 8 --mixing-horizon 27 --save-dir cifar_runs/3b_prelu_h27"
B2[6]="--n-blocks 2 --modulus-type complex_modulus --L 8 --global-avg-pool --save-dir cifar_runs/2b_cmod_gap"

run_batch "batch2" B2

# =========================================================================
# Batch 3: Data efficiency sweeps (complex_modulus + depth variations)
# =========================================================================
declare -A B3
B3[0]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.1 --save-dir cifar_runs/3b_cmod_s01"
B3[1]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.3 --save-dir cifar_runs/3b_cmod_s03"
B3[2]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.5 --save-dir cifar_runs/3b_cmod_s05"
B3[3]="--n-blocks 2 --modulus-type phase_relu --L 8 --train-size 0.1 --save-dir cifar_runs/2b_prelu_s01"
B3[4]="--n-blocks 2 --modulus-type phase_relu --L 8 --train-size 0.3 --save-dir cifar_runs/2b_prelu_s03"
B3[5]="--n-blocks 4 --modulus-type phase_relu --L 8 --lowpass-last --mixing-horizon 27 --kernel-size 5 --train-size 0.1 --save-dir cifar_runs/4b_prelu_lp_s01"
B3[6]="--n-blocks 4 --modulus-type phase_relu --L 8 --lowpass-last --mixing-horizon 27 --kernel-size 5 --train-size 0.3 --save-dir cifar_runs/4b_prelu_lp_s03"

run_batch "batch3" B3

echo ""
echo "========================================"
echo "  All 3 batches complete."
echo "========================================"
