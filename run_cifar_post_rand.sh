#!/bin/bash
# run_cifar_post_rand.sh — Random-init post-experiments (70 per dataset)
# Usage: ./run_cifar_post_rand.sh <num_gpus> <dataset>
#   dataset: cifar10 or cifar100
set -euo pipefail

NGPUS=${1:?Usage: $0 <num_gpus> <dataset>}
DATASET=${2:?Usage: $0 <num_gpus> <dataset>}

if [[ "$DATASET" != "cifar10" && "$DATASET" != "cifar100" ]]; then
    echo "Error: dataset must be cifar10 or cifar100"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate scatter_net

COMMON="--dataset $DATASET --random-init --joint --epochs 180 --global-batch-size 128 --print-freq 200 --workers 10 --kernel-size 5"
BASE_DIR="cifar_post_${DATASET}"
LOG_DIR="logs_post_rand_${DATASET}"
mkdir -p "$LOG_DIR"

JOBS=()

# 3b_cmod_4L
for TS in 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do
  TS_TAG=$(echo "$TS" | sed 's/0\.//;s/1\.0/10/')
  for SEED in 42 43 44 45 46; do
    SAVE="${BASE_DIR}/3b_cmod_4L_rand_s${TS_TAG}_seed${SEED}"
    if [ "$TS" = "1.0" ]; then
      TS_ARG=""
    else
      TS_ARG="--train-size $TS"
    fi
    JOBS+=("--n-blocks 3 --modulus-type complex_modulus --L 4 $TS_ARG --seed $SEED --save-dir $SAVE")
  done
done

# 4b_cmod_4L_h27
for TS in 0.05 0.1 0.15 0.2 0.3 0.5 1.0; do
  TS_TAG=$(echo "$TS" | sed 's/0\.//;s/1\.0/10/')
  for SEED in 42 43 44 45 46; do
    SAVE="${BASE_DIR}/4b_cmod_4L_h27_rand_s${TS_TAG}_seed${SEED}"
    if [ "$TS" = "1.0" ]; then
      TS_ARG=""
    else
      TS_ARG="--train-size $TS"
    fi
    JOBS+=("--n-blocks 4 --modulus-type complex_modulus --L 4 --lowpass-last --mixing-horizon 27 $TS_ARG --seed $SEED --save-dir $SAVE")
  done
done

TOTAL=${#JOBS[@]}
NBATCHES=$(( (TOTAL + NGPUS - 1) / NGPUS ))
echo "Random-init post ($DATASET): $TOTAL experiments, $NGPUS GPUs, $NBATCHES batches"
echo ""

FAILED_TOTAL=0
BATCH=0

for ((i=0; i<TOTAL; i+=NGPUS)); do
    BATCH=$((BATCH + 1))
    PIDS=()
    DIRS=()
    echo "======== Batch $BATCH/$NBATCHES ========"

    for ((g=0; g<NGPUS && i+g<TOTAL; g++)); do
        IDX=$((i + g))
        SAVE_DIR=$(echo "${JOBS[$IDX]}" | grep -oP '(?<=--save-dir )\S+')
        DIRS+=("$SAVE_DIR")
        if ls "$SAVE_DIR"/results_*.json 1>/dev/null 2>&1; then
            echo "  [GPU $g] SKIP (results exist): $SAVE_DIR"
            continue
        fi
        echo "  [GPU $g] ${JOBS[$IDX]}"
        CUDA_VISIBLE_DEVICES=$g python train.py $COMMON ${JOBS[$IDX]} \
            > "$LOG_DIR/job${IDX}_gpu${g}.log" 2>&1 &
        PIDS+=($!)
    done

    FAILED=0
    for j in "${!PIDS[@]}"; do
        if wait "${PIDS[$j]}"; then
            echo "  [GPU $j] Done (success)"
        else
            echo "  [GPU $j] FAILED (exit code $?)"
            FAILED=$((FAILED + 1))
        fi
    done
    FAILED_TOTAL=$((FAILED_TOTAL + FAILED))

    for d in "${DIRS[@]}"; do
        if [ -d "$d" ]; then
            find "$d" -name "*.pth" -delete 2>/dev/null || true
        fi
    done

    echo "Batch $BATCH/$NBATCHES complete ($FAILED failed)"
    echo ""
done

echo "========================================"
echo "All done. $FAILED_TOTAL total failure(s) out of $TOTAL experiments."
echo "========================================"
