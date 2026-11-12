#!/bin/bash

# ================= Configuration =================
REPO_URL="git@github.com:CoenBuller/SADRL.git" 
REPO_DIR="/local/s2631415/SADRL"
LOG_FILE="active_deployments.log"

EXPERIMENTS=(
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:true:true\" ENV_NAME=clean_up N_AGENTS=7 NUM_STEPS=100"
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:false:true\" ENV_NAME=clean_up N_AGENTS=7 NUM_STEPS=100"
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:true:true\" ENV_NAME=gift N_AGENTS=5 NUM_STEPS=100"
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:false:true\" ENV_NAME=gift N_AGENTS=5 NUM_STEPS=100"
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:true:true\" ENV_NAME=pd_arena N_AGENTS=4 NUM_STEPS=25"
    "SEEDS=\"30 31 32\" VARIANTS=\"RecurrentIPPO:false:true\" ENV_NAME=pd_arena N_AGENTS=4 NUM_STEPS=25"
)

PREFIX="00650"
# =================================================

NODES=()
for i in {0..100}; do
    NODES+=("$(printf "%s%02d" "$PREFIX" "$i")")
done

echo "Starting deployment of ${#EXPERIMENTS[@]} experiments..."
echo "Logging successful deployments to: $LOG_FILE"
echo "-------------------------------------------------" > "$LOG_FILE"
echo "Deployment Run: $(date)" >> "$LOG_FILE"
echo "-------------------------------------------------" >> "$LOG_FILE"

printf "%-15s | %s\n" "NODE" "ACTION"
printf "%-15s-+-%s\n" "----------------" "---------------------------------------"

EXP_IDX=0
NODE_IDX=0
TOTAL_NODES=${#NODES[@]}
DEPLOYED_LOG=()

while [ $EXP_IDX -lt ${#EXPERIMENTS[@]} ]; do
    if [ $NODE_IDX -ge $TOTAL_NODES ]; then
        echo "No more free nodes available. Waiting 2 minutes before retrying..."
        sleep 120
        NODE_IDX=0
        continue
    fi

    NODE="${NODES[$NODE_IDX]}"
    EXP_PARAMS="${EXPERIMENTS[$EXP_IDX]}"
    
    ssh -q -T -o LogLevel=QUIET -o BatchMode=yes -o ConnectTimeout=5 "$NODE" '
        REMOTE_USER=$(whoami)
        OTHER_USERS=$(who | awk "{print \$1}" | sort -u | grep -v "^${REMOTE_USER}$" | wc -l)
        if [ "$OTHER_USERS" -gt 0 ]; then exit 1; fi
        
        if ! command -v nvidia-smi &> /dev/null; then exit 3; fi
        
        GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
        if [ -n "$GPU_PIDS" ]; then exit 2; fi
        
        exit 0
    ' > /dev/null 2>&1
    
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        printf "%-15s | \e[32m[DEPLOYING]\e[0m Experiment $((EXP_IDX+1)) -> %s\n" "$NODE" "$EXP_PARAMS"
        
        ssh -q -T -o BatchMode=yes "$NODE" "
            mkdir -p /local/s2631415 && chmod 700 /local/s2631415

            if [ ! -d \"$REPO_DIR\" ]; then
                cd /local/s2631415
                git clone $REPO_URL
            fi
            
            cd \"$REPO_DIR\"
            git checkout coen
            git pull
            
            tmux new-session -d -s train 'source ~/.bashrc && $EXP_PARAMS ./run_local.sh'
        "
        
        # Log to array for terminal summary, and write directly to file
        DEPLOYED_LOG+=("[$NODE] running: $EXP_PARAMS")
        echo "[$NODE] running: $EXP_PARAMS" >> "$LOG_FILE"
        
        EXP_IDX=$((EXP_IDX + 1))
    fi
    
    NODE_IDX=$((NODE_IDX + 1))
done

echo ""
echo "================================================="
echo "All ${#EXPERIMENTS[@]} experiments successfully deployed!"
echo "Deployment Summary (Saved to $LOG_FILE):"
echo "-------------------------------------------------"
for entry in "${DEPLOYED_LOG[@]}"; do
    echo "$entry"
done
echo "================================================="