#!/bin/bash
# run_cifar.sh — Scattering-init experiments (144 total)
# Usage: ./run_cifar.sh <num_gpus>
set -euo pipefail

NGPUS=${1:?Usage: $0 <num_gpus>}

eval "$(conda shell.bash hook)"
conda activate scatter_net

COMMON="--lr-epochs 20 --epochs 100 --global-batch-size 128 --print-freq 200 --workers 10 --kernel-size 5"
BASE_DIR="cifar_exps"
LOG_DIR="logs_scat"
mkdir -p "$LOG_DIR"

JOBS=()

for BLOCKS in 3 4; do
  for MOD in phase_relu complex_modulus; do
    MOD_SHORT=$( [ "$MOD" = "phase_relu" ] && echo "prelu" || echo "cmod" )
    for L in 4 6 8; do

      if [ "$BLOCKS" -eq 3 ]; then
        HORIZONS=("none")
      else
        HORIZONS=(27 243)
      fi

      for H in "${HORIZONS[@]}"; do
        ARCH_ARGS="--n-blocks $BLOCKS --modulus-type $MOD --L $L"

        if [ "$BLOCKS" -eq 4 ]; then
          ARCH_ARGS="$ARCH_ARGS --lowpass-last --mixing-horizon $H"
          ARCH_TAG="${BLOCKS}b_${MOD_SHORT}_${L}L_h${H}"
        else
          ARCH_TAG="${BLOCKS}b_${MOD_SHORT}_${L}L"
        fi

        for TS in 0.1 0.3 0.5 1.0; do
          TS_TAG=$(echo "$TS" | sed 's/0\.//;s/1\.0/10/')
          for L2 in 0 0.005; do
            if [ "$L2" = "0" ]; then
              L2_TAG=""
              L2_ARG=""
            else
              L2_TAG="_l2"
              L2_ARG="--l2-penalty $L2"
            fi
            SAVE="${BASE_DIR}/${ARCH_TAG}_s${TS_TAG}${L2_TAG}"
            if [ "$TS" = "1.0" ]; then
              TS_ARG=""
            else
              TS_ARG="--train-size $TS"
            fi
            JOBS+=("$ARCH_ARGS $TS_ARG $L2_ARG --save-dir $SAVE")
          done
        done
      done
    done
  done
done

TOTAL=${#JOBS[@]}
NBATCHES=$(( (TOTAL + NGPUS - 1) / NGPUS ))
echo "Scattering-init: $TOTAL experiments, $NGPUS GPUs, $NBATCHES batches"
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
        # Skip if results already exist (resumable)
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

    # Checkpoint cleanup: keep only results_*.json
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
