#!/bin/sh
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode[13-14]
#SBATCH --exclude=gnode01
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --export=ALL




srun python collect_pick_place.py \
    --env_name UR5e_PickPlaceDistractor \
    --ctrl_config OSC_POSE \
    --object_set 2 \
    --task_id 0 \
    --camera_obs