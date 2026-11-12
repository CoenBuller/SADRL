#!/bin/bash
# Run MAPPO, RecurrentMAPPO, and RecurrentIPPO variants locally.
#
# Usage:
#   ./run_local.sh                                       # default variants (incl. RecurrentIPPO w/ LSTM)
#   SEEDS="30" ./run_local.sh                            # single seed
#   SMOKE=1 ./run_local.sh                               # run smoke_test.py first, abort if it fails
#   TOTAL_TIMESTEPS=1000000 WANDB=false ./run_local.sh   # quick local run
#   VARIANTS="RecurrentIPPO:false:true" ./run_local.sh   # only RecurrentIPPO with LSTM=true
#   LSTM=true ./run_local.sh                             # force LSTM=true across all variants
#   PARALLEL=1 ./run_local.sh                            # all runs at once (needs enough GPU memory)
#
# Variants can be "<ALGO>:<PREDICT_POS>" or "<ALGO>:<PREDICT_POS>:<LSTM>".

set -uo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-experiments/recurrent.yaml}"
ENV_NAME="${ENV_NAME:-clean_up}"
SEEDS="${SEEDS:-30 31 32}"
VARIANTS="${VARIANTS:-MAPPO:false:false RecurrentMAPPO:false:false RecurrentMAPPO:true:false RecurrentIPPO:false:true RecurrentIPPO:true:true}"
WANDB="${WANDB:-true}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-}"      # empty = use the value from the YAML
PARALLEL="${PARALLEL:-0}"
SMOKE="${SMOKE:-0}"
LOG_DIR="${LOG_DIR:-local_logs}"
PYTHON="${PYTHON:-python}"
N_AGENTS="${N_AGENTS:-7}"
NUM_STEPS="${NUM_STEPS:-}"                  # empty = use the value from the YAML
LSTM="${LSTM:-}"                            # global override; empty = use per-variant or YAML value

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR"

if [ "$SMOKE" = "1" ]; then
    echo "Running smoke test first..."
    uv run smoke_test.py --env "$ENV_NAME" || { echo "Smoke test failed, aborting."; exit 1; }
fi

pids=()
names=()
failed=0

run_one() {
    local algo="$1" pos="$2" lstm_val="$3" seed="$4"
    local name="${algo}_pos_${pos}_lstm_${lstm_val:-default}_seed${seed}"
    local log="$LOG_DIR/${name}.log"
    local extra=()
    [ -n "$TOTAL_TIMESTEPS" ] && extra+=(--total_timesteps "$TOTAL_TIMESTEPS")
    [ -n "$NUM_STEPS" ] && extra+=(--num_steps "$NUM_STEPS")
    [ -n "$lstm_val" ] && extra+=(--lstm "$lstm_val")

    echo "[$(date +%H:%M:%S)] START $name -> $log"
    uv run main.py --config "$CONFIG" --algo "$algo" --predict_pos "$pos" \
    --env "$ENV_NAME" --seed "$seed" --wandb "$WANDB" --n_agents "$N_AGENTS" "${extra[@]}" > "$log" 2>&1
}

for seed in $SEEDS; do
    for variant in $VARIANTS; do
        IFS=':' read -r algo pos var_lstm <<< "$variant"
        # Global LSTM env var overrides variant-level LSTM if set
        effective_lstm="${LSTM:-$var_lstm}"
        run_name="${algo}_pos_${pos}_lstm_${effective_lstm:-default}_seed${seed}"

        if [ "$PARALLEL" = "1" ]; then
            run_one "$algo" "$pos" "$effective_lstm" "$seed" &
            pids+=($!)
            names+=("$run_name")
        else
            if run_one "$algo" "$pos" "$effective_lstm" "$seed"; then
                echo "[$(date +%H:%M:%S)] DONE  $run_name"
            else
                echo "[$(date +%H:%M:%S)] FAIL  $run_name (see $LOG_DIR)"
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