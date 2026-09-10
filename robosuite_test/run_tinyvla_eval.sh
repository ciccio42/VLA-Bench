#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

# Runs a TinyVLA LoRA checkpoint through this repo's robosuite simulation, in-process (no
# server/client split needed -- unlike lerobot/mimic_video, TinyVLA's own deps coexist with this
# harness's conda env). Env is tinyvla_robosuite_1_0_1_provola (confirmed by earlier TinyVLA runs
# in this repo), not the generic run_robosuite_eval.sh's openvla_robosuite_1_0_1.
#
# Usage: sbatch run_tinyvla_eval.sh <task_suite_name> <otd:true|false> <model_path> <model_base> [run_number] [num_trials_per_task]

set -uo pipefail

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH

TASK_SUITE_NAME="${1:?task_suite_name required}"
OTD="${2:?otd (true/false) required}"
MODEL_PATH="${3:?model_path required}"
MODEL_BASE="${4:?model_base required}"
RUN_NUMBER="${5:-0}"
NUM_TRIALS_PER_TASK="${6:-10}"

echo "TASK_SUITE_NAME: ${TASK_SUITE_NAME}"
echo "OTD: ${OTD}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "MODEL_BASE: ${MODEL_BASE}"
echo "RUN_NUMBER: ${RUN_NUMBER}"
echo "NUM_TRIALS_PER_TASK: ${NUM_TRIALS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

srun python run_robosuite_eval.py \
    --config_path="models/tinyvla_eval_config.yml" \
    --task_suite_name "${TASK_SUITE_NAME}" \
    --run_number "${RUN_NUMBER}" \
    --change_spawn_regions false \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    --model_config.otd "${OTD}" \
    --model_config.model_path="${MODEL_PATH}" \
    --model_config.model_base="${MODEL_BASE}" \
    --model_config.task_suite_name "${TASK_SUITE_NAME}"
