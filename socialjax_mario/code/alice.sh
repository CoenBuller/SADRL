#!/usr/bin/env bash
# alice.sh: run SocialJax jobs on the ALICE cluster (Leiden University) from your own machine.
# Everything on ALICE lives under ~/socialjax (code, venv, logs, outputs) so it never mixes
# with other projects. Run `./alice.sh help` for the list of commands.
set -euo pipefail

# --- Configuration (override with environment variables) --------------------
SSH_HOST="${ALICE_HOST:-alice}"                                # Host alias in ~/.ssh/config
REMOTE_DIR="${ALICE_DIR:-socialjax}"                           # Project dir on ALICE, relative to $HOME
VENV_DIR="${ALICE_VENV:-socialjax/venv}"                       # Python venv on ALICE, relative to $HOME
PYTHON_MODULE="${ALICE_PYTHON:-Python/3.10.8-GCCcore-12.2.0}"  # requirements.txt pins target Python 3.10
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local-only artefacts are never uploaded; the remote copies of these are fetched back instead.
RSYNC_EXCLUDES=(--exclude .git/ --exclude __pycache__/ --exclude '*.pyc' --exclude .venv/ --exclude venv/
                --exclude logs/ --exclude outputs/ --exclude wandb/ --exclude checkpoints/
                --exclude results/ --exclude handoff/)

# Run a bash script (read from stdin) as a login shell on ALICE, so `module` is available.
remote() {
    ssh "$SSH_HOST" "bash -l -s -- $(printf '%q ' "$@")"
}

usage() {
    cat <<EOF
Usage: ./alice.sh <command> [args]

  check                      Test the connection; show accounts, partitions and your jobs
  setup                      Sync code and create/update the venv ~/$VENV_DIR from requirements.txt
  sync                       Copy this project to ~/$REMOTE_DIR on ALICE
  submit [opts] F.sbatch [a] Sync, then sbatch F.sbatch; options before it go to sbatch,
                             arguments after it go to the job.  Example:
                               ./alice.sh submit --job-name=sj-ippo-coins slurm/train.sbatch --algo IPPO --env coins
  gpu-test                   Submit a 10-minute job checking that JAX sees a GPU
  status                     Show your queued and running jobs
  logs <jobid> [-f]          Print (or follow, with -f) a job's log
  cancel <jobid>...          Cancel jobs
  fetch                      Copy logs/, outputs/, checkpoints/ and wandb/ from ALICE into results/alice/
  shell                      Open an interactive shell on ALICE in the project directory
EOF
}

cmd_sync() {
    echo ">> Syncing $LOCAL_DIR -> $SSH_HOST:~/$REMOTE_DIR"
    ssh "$SSH_HOST" "mkdir -p ~/$(printf '%q' "$REMOTE_DIR")/logs"
    rsync -az "${RSYNC_EXCLUDES[@]}" "$LOCAL_DIR/" "$SSH_HOST:$REMOTE_DIR/"
}

cmd_check() {
    remote <<'EOF'
echo "Connected to $(hostname) as $USER"
echo "SLURM accounts: $(sacctmgr -nP show assoc user="$USER" format=account | sort -u | paste -sd, -)"
echo
sinfo -s
echo
squeue -u "$USER"
EOF
}

cmd_setup() {
    cmd_sync
    echo ">> Creating/updating venv ~/$VENV_DIR with $PYTHON_MODULE"
    remote "$REMOTE_DIR" "$VENV_DIR" "$PYTHON_MODULE" <<'EOF'
set -eo pipefail
module load "$3"
venv="$HOME/$2"
[ -d "$venv" ] || python -m venv "$venv"
source "$venv/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r "$HOME/$1/requirements.txt"
python -c 'import jax; print("Installed JAX", jax.__version__)'
EOF
}

cmd_submit() {
    local sbatch_args=() script=""
    while [ $# -gt 0 ]; do
        sbatch_args+=("$1")
        if [[ "$1" == *.sbatch ]]; then script="$1"; shift; break; fi
        shift
    done
    [ -n "$script" ] || { echo "error: no .sbatch file given" >&2; usage; exit 1; }
    cmd_sync
    remote "$REMOTE_DIR" "$PYTHON_MODULE" "$VENV_DIR" "${sbatch_args[@]}" -- "$@" <<'EOF'
set -eo pipefail
cd "$HOME/$1"
export PYTHON_MODULE="$2" VENV_DIR="$3"
shift 3
sbatch_args=()
while [ "$1" != "--" ]; do sbatch_args+=("$1"); shift; done
shift
jobid=$(sbatch --parsable --export=ALL "${sbatch_args[@]}" "$@")
echo "Submitted job $jobid. Follow it with: ./alice.sh logs $jobid -f"
EOF
}

cmd_logs() {
    local jobid="${1:?usage: ./alice.sh logs <jobid> [-f]}" follow="${2:-}"
    remote "$REMOTE_DIR" "$jobid" "$follow" <<'EOF'
cd "$HOME/$1"
log=$(ls logs/*_"$2".out 2>/dev/null | head -1)
[ -n "$log" ] || { echo "No log for job $2 yet (still queued?)"; exit 1; }
if [ "$3" = "-f" ]; then tail -n +1 -f "$log"; else cat "$log"; fi
EOF
}

cmd_fetch() {
    echo ">> Fetching results from $SSH_HOST:~/$REMOTE_DIR into results/alice/"
    mkdir -p "$LOCAL_DIR/results/alice"
    for d in logs outputs checkpoints wandb; do
        rsync -az "$SSH_HOST:$REMOTE_DIR/$d" "$LOCAL_DIR/results/alice/" 2>/dev/null || true
    done
}

case "${1:-help}" in
    check)    cmd_check ;;
    setup)    cmd_setup ;;
    sync)     cmd_sync ;;
    submit)   shift; cmd_submit "$@" ;;
    gpu-test) cmd_submit --job-name=sj-gpu-check --time=00:10:00 slurm/python.sbatch slurm/gpu_check.py ;;
    status)   ssh "$SSH_HOST" 'squeue -u "$USER" -o "%.10i %.12P %.20j %.8T %.10M %.10l %R"' ;;
    logs)     shift; cmd_logs "$@" ;;
    cancel)   shift; ssh "$SSH_HOST" "scancel $(printf '%q ' "$@")" ;;
    fetch)    cmd_fetch ;;
    shell)    ssh -t "$SSH_HOST" "cd ~/$(printf '%q' "$REMOTE_DIR") 2>/dev/null; exec bash -l" ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $1" >&2; usage; exit 1 ;;
esac
