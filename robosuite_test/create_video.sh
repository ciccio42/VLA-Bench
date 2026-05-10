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

export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210" #"/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/frosa_Loc/.mujoco/mujoco210/bin" #/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
PKL_PATH=/user/frosa/Multi-Task-LFD-Framework/repo/mimic-play/MimicPlay/trained_models_lowlevel_demo_human_agent_ur5e_3D_state_same_conf_with_valid_modes_5/rollout_pick_place_0_False_obj_set_-1_change_command_False_epoch_170

python create_video.py \
    --path_to_pkl $PKL_PATH \
    --output_dir $PKL_PATH/video \
    #--debug
