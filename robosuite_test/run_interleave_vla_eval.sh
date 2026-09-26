#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --qos=did_robot_learning_359_gpuq_qos
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

# Runs a trained Interleave-pi0 checkpoint (see Interleave-VLA/open-pi-zero) through this
# repo's robosuite simulation. Mirrors run_lerobot_eval.sh / run_mimic_video_eval.sh.
#
# The policy and the sim run in two different conda envs (Interleave-pi0 needs torch 2.5.1 +
# tensorflow, incompatible with this harness's own env), so this script starts a small
# Interleave-pi0 inference server in the `pi0` env, waits for it to become healthy, then runs
# the usual robosuite client against it in this repo's own env.
#
# Usage: sbatch run_interleave_vla_eval.sh [config_yml] [checkpoint_path] [train_config_path] [dataset_statistics_path] [port] [run_number] [num_trials_per_task] [task_suite_name] [otd] [object_set]

set -uo pipefail

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

INTERLEAVE_VLA_DIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Interleave-VLA/open-pi-zero

CONFIG=${1:-models/interleave_vla_eval_config.yml}
CHECKPOINT_PATH=${2:?"checkpoint_path required, e.g. ${INTERLEAVE_VLA_DIR}/logs/train/<run>/checkpoint/step12345.pt"}
TRAIN_CONFIG_PATH=${3:?"train_config_path required, e.g. ${INTERLEAVE_VLA_DIR}/config/train/interleaved_ur5e_sim_grounding_bin_rm_0_5_10_15.yaml"}
DATASET_STATISTICS_PATH=${4:?"dataset_statistics_path required, the dataset_statistics_*.json next to the TFDS dataset the checkpoint trained on"}
PORT=${5:-8766}
RUN_NUMBER=${6:-0}
NUM_TRIALS_PER_TASK=${7:-1}
TASK_SUITE_NAME=${8:-ur5e_pick_place_delta_all}
OTD=${9:-false}
OBJECT_SET=${10:--1}
CHANGE_SPAWN_REGIONS=${CHANGE_SPAWN_REGIONS:-false}
DEBUG_SAVE_IMAGES=${DEBUG_SAVE_IMAGES:-false}
DEBUG_IMAGE_DIR=${DEBUG_IMAGE_DIR:-./debug_interleave_images}
# Number of predicted steps the interleave controller executes open-loop before re-querying
# the server (<= the model's horizon_steps=4). Defaults to the InterleaveVLAConfig default
# (3); override to test e.g. chunk_size=1 (re-query every env step, fully closed-loop).
CHUNK_SIZE=${CHUNK_SIZE:-3}
# "auto" reads action_proprio_normalization_type from TRAIN_CONFIG_PATH -- safe for any
# checkpoint trained after the interleaved_dataset.py propagation fix. Override to "bounds"
# explicitly for the 5 pre-fix UR5e sim grounding-bin checkpoints (all actually trained with
# BOUNDS regardless of what their YAML said before 2026-09-21).
NORMALIZATION_TYPE=${NORMALIZATION_TYPE:-auto}

USE_COSMOS_NAME=${USE_COSMOS_NAME:-false}
MODEL_COSMOS_NAME=${MODEL_COSMOS_NAME:-nvidia/Cosmos-Reason2-8B}
MODEL_COSMOS_PORT=${MODEL_COSMOS_PORT:-8000}
DATASET_PATH=${DATASET_PATH:-/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset}

echo "CONFIG: ${CONFIG}"
echo "CHECKPOINT_PATH: ${CHECKPOINT_PATH}"
echo "TRAIN_CONFIG_PATH: ${TRAIN_CONFIG_PATH}"
echo "DATASET_STATISTICS_PATH: ${DATASET_STATISTICS_PATH}"
echo "PORT: ${PORT}"
echo "RUN_NUMBER: ${RUN_NUMBER}"
echo "NUM_TRIALS_PER_TASK: ${NUM_TRIALS_PER_TASK}"
echo "TASK_SUITE_NAME: ${TASK_SUITE_NAME}"
echo "CHANGE_SPAWN_REGIONS: ${CHANGE_SPAWN_REGIONS}"
echo "NORMALIZATION_TYPE: ${NORMALIZATION_TYPE}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# --- start the Interleave-pi0 inference server in the pi0 conda env ---
PI0_CONDA=/mnt/beegfs/frosa/.conda/envs/pi0
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${INTERLEAVE_VLA_DIR}/checkpoints/"}

(cd "${INTERLEAVE_VLA_DIR}" && "${PI0_CONDA}/bin/python" scripts/interleave_vla_policy_server.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --train_config_path "${TRAIN_CONFIG_PATH}" \
    --dataset_statistics_path "${DATASET_STATISTICS_PATH}" \
    --normalization_type "${NORMALIZATION_TYPE}" \
    --port "${PORT}") &
SERVER_PID=$!

echo "Waiting for Interleave-pi0 policy server (pid ${SERVER_PID}) on port ${PORT}..."
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

# --- run the robosuite client in the benchmark's own conda env ---
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH
COSMOS_ARGS=()
if [ "${USE_COSMOS_NAME}" = "true" ]; then
    COSMOS_ARGS=(
        --model_config.use_cosmos_name=true
        --model_config.model_cosmos_name="${MODEL_COSMOS_NAME}"
        --model_config.model_cosmos_port="${MODEL_COSMOS_PORT}"
        --model_config.dataset_path="${DATASET_PATH}"
    )
fi
# run_robosuite_eval.py treats model_config.model_path as a DIRECTORY (it creates
# rollout_*/ subfolders under it), matching lerobot/mimic_video's convention of a
# checkpoint directory rather than a single file. Derive one per checkpoint step so
# rollouts from different steps of the same run don't collide.
CKPT_BASENAME=$(basename "${CHECKPOINT_PATH}" .pt)
EVAL_SAVE_DIR="$(dirname "${CHECKPOINT_PATH}")/${CKPT_BASENAME}_eval"
mkdir -p "${EVAL_SAVE_DIR}"

srun python run_robosuite_eval.py \
    --config_path="${CONFIG}" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --run_number "${RUN_NUMBER}" \
    --change_spawn_regions "${CHANGE_SPAWN_REGIONS}" \
    --object_set "${OBJECT_SET}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --model_config.model_path="${EVAL_SAVE_DIR}" \
    --model_config.server_port="${PORT}" \
    --model_config.task_suite_name="${TASK_SUITE_NAME}" \
    --model_config.otd="${OTD}" \
    --model_config.debug_save_images="${DEBUG_SAVE_IMAGES}" \
    --model_config.debug_image_dir="${DEBUG_IMAGE_DIR}" \
    --model_config.chunk_size="${CHUNK_SIZE}" \
    "${COSMOS_ARGS[@]}"
CLIENT_EXIT=$?

kill "${SERVER_PID}" 2>/dev/null
wait "${SERVER_PID}" 2>/dev/null
exit ${CLIENT_EXIT}
