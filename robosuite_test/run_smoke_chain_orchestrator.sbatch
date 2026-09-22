#!/bin/bash
#SBATCH --job-name=smoke-chain-orchestrator
#SBATCH --output=logs/orchestrator_%j.out
#SBATCH --error=logs/orchestrator_%j.err
#SBATCH --time=06:45:00
#SBATCH --partition=aiq
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --export=ALL

# Runs run_smoke_chain_with_telegram.sh as a standalone SLURM job, independent of any
# login session or agent Monitor timeout (an earlier attempt ran this same logic inside
# a Monitor tool call, which killed the process when the Monitor's 30-minute window
# expired, silently dropping 2 of 4 queued smoke tests -- this sbatch wrapper follows
# the same pattern as slurm/run_autoresume_watcher.sh specifically to avoid that).
set -uo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
bash run_smoke_chain_with_telegram.sh
