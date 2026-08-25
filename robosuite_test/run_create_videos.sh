#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

# One-off: render mp4 videos from smoke-test rollout pkls. Needs a compute node — unpickling a
# traj touches robosuite_utils.py, which imports robosuite -> mujoco_py, and mujoco_py JIT-builds
# a native extension on first import; that build only succeeds with the mujoco/EGL env below and
# apparently only on the gnode compute nodes (login-node gcc/ld are incompatible).

set -uo pipefail

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

CONDA=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin/python
OUT=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test/lerobot_eval_videos

MDIR=/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_molmoact2/checkpoints/040500/pretrained_model/rollout_pick_place_3_False_obj_set_-1_change_command_False_use_vllm_False_otd_False

mkdir -p "$OUT/molmoact2_040500_run3"

echo "=== MolmoAct2 videos (step 39500, run 0) ==="
srun "$CONDA" create_video.py --path_to_pkl "$MDIR" --output_dir "$OUT/molmoact2_040500_run3"

echo "DONE"
