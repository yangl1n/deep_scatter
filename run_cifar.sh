#!/bin/bash
# run_all.sh — 7 experiments in parallel, one per GPU
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate scatter_net

COMMON="--lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 200"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

declare -A EXPS

EXPS[0]="--n-blocks 3 --modulus-type phase_relu --L 8 --save-dir cifar_runs/3b_prelu"
EXPS[1]="--n-blocks 2 --L 8 --save-dir cifar_runs/2b_cmod"
EXPS[2]="--n-blocks 3 --modulus-type complex_modulus --L 8 --save-dir cifar_runs/3b_cmod"
EXPS[3]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.1 --save-dir cifar_runs/3b_prelu_s01"
EXPS[4]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.3 --save-dir cifar_runs/3b_prelu_s03"
EXPS[5]="--n-blocks 3 --modulus-type phase_relu --L 8 --train-size 0.5 --save-dir cifar_runs/3b_prelu_s05"
EXPS[6]="--n-blocks 4 --modulus-type phase_relu --lowpass-last --mixing-horizon 27 --kernel-size 5 --save-dir cifar_runs/4b_prelu_lp"

PIDS=()

for GPU in "${!EXPS[@]}"; do
    echo "[GPU $GPU] Starting: ${EXPS[$GPU]}"
    CUDA_VISIBLE_DEVICES=$GPU python train.py $COMMON ${EXPS[$GPU]} \
        > "$LOG_DIR/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 7 experiments launched. PIDs: ${PIDS[*]}"
echo "Logs: $LOG_DIR/gpu{0..6}.log"
echo ""

# Wait for all and report results
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[GPU $i] Done (success)"
    else
        echo "[GPU $i] FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "All experiments completed successfully."
else
    echo "$FAILED experiment(s) failed. Check logs."
fi
