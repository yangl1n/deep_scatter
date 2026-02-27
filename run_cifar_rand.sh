#!/bin/bash
# run_cifar_rand.sh — Random-init baselines: 12 experiments in 2 batches
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scatter_net

COMMON="--random-init --joint --epochs 120 --global-batch-size 128 --print-freq 200"
LOG_DIR="logs_rand"
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
# Batch 1: Full-data architecture grid (mirrors scattering baselines)
# =========================================================================
declare -A B1
B1[0]="--n-blocks 3 --modulus-type phase_relu --L 8 --save-dir cifar_runs/3b_prelu_rand"
B1[1]="--n-blocks 2 --L 8 --save-dir cifar_runs/2b_cmod_rand"
B1[2]="--n-blocks 3 --modulus-type complex_modulus --L 8 --save-dir cifar_runs/3b_cmod_rand"
B1[3]="--n-blocks 2 --modulus-type phase_relu --L 8 --save-dir cifar_runs/2b_prelu_rand"
B1[4]="--n-blocks 4 --modulus-type phase_relu --lowpass-last --mixing-horizon 27 --kernel-size 5 --save-dir cifar_runs/4b_prelu_lp_rand"
B1[5]="--n-blocks 4 --modulus-type complex_modulus --L 8 --lowpass-last --mixing-horizon 27 --kernel-size 5 --save-dir cifar_runs/4b_cmod_lp_rand"

run_batch "rand_full" B1

# =========================================================================
# Batch 2: Data-efficiency sweeps (3b_prelu + 3b_cmod at 10%/30%/50%)
# =========================================================================
declare -A B2
B2[0]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.1 --save-dir cifar_runs/3b_prelu_rand_s01"
B2[1]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.3 --save-dir cifar_runs/3b_prelu_rand_s03"
B2[2]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.5 --save-dir cifar_runs/3b_prelu_rand_s05"
B2[3]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.1 --save-dir cifar_runs/3b_cmod_rand_s01"
B2[4]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.3 --save-dir cifar_runs/3b_cmod_rand_s03"
B2[5]="--n-blocks 3 --modulus-type complex_modulus --L 8 --train-size 0.5 --save-dir cifar_runs/3b_cmod_rand_s05"

run_batch "rand_data" B2

echo ""
echo "========================================"
echo "  All random-init batches complete."
echo "========================================"
