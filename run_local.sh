#!/bin/bash
# Run MAPPO, RecurrentMAPPO and RecurrentMAPPO+position-prediction locally.
#
# Usage:
#   ./run_local.sh                          # all 3 variants, seeds 30 31, sequentially
#   SEEDS="30" ./run_local.sh               # single seed
#   SMOKE=1 ./run_local.sh                  # run smoke_test.py first, abort if it fails
#   TOTAL_TIMESTEPS=1000000 WANDB=false ./run_local.sh   # quick local run
#   VARIANTS="RecurrentMAPPO:true" ./run_local.sh        # only the position run
#   PARALLEL=1 ./run_local.sh               # all runs at once (needs enough GPU memory)
#
# Variants are "<ALGO>:<PREDICT_POS>".

set -uo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-experiments/recurrent.yaml}"
ENV_NAME="${ENV_NAME:-clean_up}"
SEEDS="${SEEDS:-30 31 32}"
VARIANTS="${VARIANTS:-MAPPO:false RecurrentMAPPO:false RecurrentMAPPO:true}"
WANDB="${WANDB:-true}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"   # empty = use the value from the YAML
PARALLEL="${PARALLEL:-0}"
SMOKE="${SMOKE:-0}"
LOG_DIR="${LOG_DIR:-local_logs}"
PYTHON="${PYTHON:-python}"
N_AGENTS="${N_AGENTS:-7}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

if [ "$SMOKE" = "1" ]; then
    echo "Running smoke test first..."
    "$PYTHON" smoke_test.py --env "$ENV_NAME" || { echo "Smoke test failed, aborting."; exit 1; }
fi

pids=()
names=()
failed=0

run_one() {
    local algo="$1" pos="$2" seed="$3"
    local name="${algo}_pos_${pos}_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    local extra=()
    [ -n "$TOTAL_TIMESTEPS" ] && extra+=(--total_timesteps "$TOTAL_TIMESTEPS")

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    "$PYTHON" main.py --config "$CONFIG" --algo "$algo" --predict_pos "$pos" \
        --env "$ENV_NAME" --seed "$seed" --wandb "$WANDB" --n_agents "$N_AGENTS" "${extra[@]}" > "$log" 2>&1
}

for seed in $SEEDS; do
    for variant in $VARIANTS; do
        algo="${variant%%:*}"
        pos="${variant##*:}"
        if [ "$PARALLEL" = "1" ]; then
            run_one "$algo" "$pos" "$seed" &
            pids+=($!)
            names+=("${algo}_pos_${pos}_seed${seed}")
        else
            if run_one "$algo" "$pos" "$seed"; then
                echo "[$(date +%H:%M:%S)] DONE  ${algo}_pos_${pos}_seed${seed}"
            else
                echo "[$(date +%H:%M:%S)] FAIL  ${algo}_pos_${pos}_seed${seed} (see $LOG_DIR)"
                failed=$((failed + 1))
            fi
        fi
    done
done

if [ "$PARALLEL" = "1" ]; then
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "DONE  ${names[$i]}"
        else
            echo "FAIL  ${names[$i]} (see $LOG_DIR)"
            failed=$((failed + 1))
        fi
    done
fi

echo "Finished with $failed failed run(s)."
exit $((failed > 0))