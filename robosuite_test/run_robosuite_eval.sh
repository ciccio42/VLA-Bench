#!/bin/bash
#SBATCH -A hpc_default
#SBATCH --exclude=tnode[01-17]
#SBATCH --exclude=gnode14
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# ur5e_pick_place_delta_all 
# ur5e_pick_place_delta_removed_0_5_10_15
# ur5e_pick_place_removed_spawn_regions
# ur5e_pick_place_rm_one_spawn
# ur5e_pick_place_rm_central_spawn

RUN_ID=$1
CHANGE_SPAWN_REGIONS=$2
echo Running evaluation for run ${RUN_ID} with change_spawn_regions: ${CHANGE_SPAWN_REGIONS}
srun torchrun --standalone --nnodes 1 --nproc-per-node 1 run_robosuite_eval.py \
    --config_path="models/tinyvla_eval_config.yml" \
    --task_suite_name "ur5e_pick_place_delta_removed_0_5_10_15" \
    --run_number ${RUN_ID} \
    --change_spawn_regions ${CHANGE_SPAWN_REGIONS} \

