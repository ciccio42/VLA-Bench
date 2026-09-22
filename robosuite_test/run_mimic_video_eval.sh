#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --export=ALL

# Runs the mimic-video world2action model through this repo's robosuite simulation. Same
# process-bridge pattern as run_lerobot_eval.sh (see that file's comment): the policy and the sim
# run in two different, mutually incompatible envs (mimic-video needs Python 3.10 +
# cosmos_predict2/imaginaire/megatron-core; this harness pins Python 3.9 + vanilla robosuite), so
# this script starts a small inference server in mimic-video's own venv, waits for it to become
# healthy, then runs the usual robosuite client against it in this repo's own env. 2 GPUs
# requested (vs. run_lerobot_eval.sh's 1) since this server loads two large models at once (the
# frozen video2world backbone + the world2action DiT) -- drop to 1 if that turns out to fit.
#
# Usage: sbatch run_mimic_video_eval.sh [config_yml] [checkpoint_dir] [port] [run_number] [num_trials_per_task]

set -uo pipefail

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CONFIG=${1:-models/mimic_video_eval_config.yml}
CHECKPOINT_DIR=${2:-/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/mimic-video/model/checkpoints/vam/ur5e/w2a_ur5e_pick_place_lr1.000e-04_layer20_bsz16_rawscale_train}
PORT=${3:-8767}
RUN_NUMBER=${4:-1}
NUM_TRIALS_PER_TASK=${5:-10}

echo "CONFIG: ${CONFIG}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "PORT: ${PORT}"
echo "RUN_NUMBER: ${RUN_NUMBER}"
echo "NUM_TRIALS_PER_TASK: ${NUM_TRIALS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# --- start the mimic-video inference server in mimic-video's own venv ---
MIMIC_VIDEO_BASE=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/mimic-video
export UR5E_VIDEO_DATA_DIR="${MIMIC_VIDEO_BASE}/dataset/ur5e_pick_place_video"
export UR5E_ACTION_DATA_DIR="${MIMIC_VIDEO_BASE}/dataset/ur5e_pick_place_action"

(
    source "${MIMIC_VIDEO_BASE}/model/.venv/bin/activate"
    cd "${MIMIC_VIDEO_BASE}/model"
    python -m scripts.mimic_video_policy_server --checkpoint_dir "${CHECKPOINT_DIR}" --port "${PORT}"
) &
SERVER_PID=$!

echo "Waiting for mimic-video policy server (pid ${SERVER_PID}) on port ${PORT}..."
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
    sleep 10
done
if [ "${UP}" -ne 1 ]; then
    echo "Server never became healthy after 10 minutes, aborting."
    kill "${SERVER_PID}" 2>/dev/null
    exit 1
fi
echo "Server is healthy."

# --- run the robosuite client in the benchmark's (Python 3.9) env ---
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH
srun --gres=gpu:1 python run_robosuite_eval.py \
    --config_path="${CONFIG}" \
    --task_suite_name "ur5e_pick_place_delta_all" \
    --run_number "${RUN_NUMBER}" \
    --change_spawn_regions false \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --model_config.model_path="${CHECKPOINT_DIR}" \
    --model_config.server_port="${PORT}"
CLIENT_EXIT=$?

kill "${SERVER_PID}" 2>/dev/null
wait "${SERVER_PID}" 2>/dev/null
exit ${CLIENT_EXIT}
