#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

# Runs a LeRobot policy checkpoint (MolmoAct2, VLA-JEPA, ...) through this repo's robosuite
# simulation. See LEROBOT_EVAL.md for the full plan and rationale.
#
# The policy and the sim run in two different conda envs (LeRobot needs Python 3.12+; this
# harness pins Python 3.9), so this script starts a small LeRobot inference server in the
# `lerobot` env, waits for it to become healthy, then runs the usual robosuite client against
# it in this repo's own env.
#
# Usage: sbatch run_lerobot_eval.sh [config_yml] [policy_path] [port]

set -uo pipefail

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CONFIG=${1:-models/lerobot_molmoact2_eval_config.yml}
POLICY_PATH=${2:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/last/pretrained_model}
PORT=${3:-8765}
RUN_NUMBER=${RUN_NUMBER:-0}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-10}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# --- start the LeRobot inference server in the lerobot (Python 3.12) env ---
LEROBOT_CONDA=/mnt/beegfs/frosa/.conda/envs/lerobot
export HF_HOME=${HF_HOME:-/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"${HF_HOME}/hub"}

"${LEROBOT_CONDA}/bin/python" \
    /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/run_eval_scripts/lerobot_policy_server.py \
    --policy_path "${POLICY_PATH}" --port "${PORT}" &
SERVER_PID=$!

echo "Waiting for LeRobot policy server (pid ${SERVER_PID}) on port ${PORT}..."
UP=0
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
        UP=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Server process died before becoming healthy, aborting."
        exit 1
    fi
    sleep 5
done
if [ "${UP}" -ne 1 ]; then
    echo "Server never became healthy after 5 minutes, aborting."
    kill "${SERVER_PID}" 2>/dev/null
    exit 1
fi
echo "Server is healthy."

# --- run the robosuite client in the benchmark's (Python 3.9) env ---
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH
srun python run_robosuite_eval.py \
    --config_path="${CONFIG}" \
    --task_suite_name "ur5e_pick_place_delta_all" \
    --run_number "${RUN_NUMBER}" \
    --change_spawn_regions false \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}"
CLIENT_EXIT=$?

kill "${SERVER_PID}" 2>/dev/null
wait "${SERVER_PID}" 2>/dev/null
exit ${CLIENT_EXIT}
