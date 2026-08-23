#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --exclude=gnode09,gnode12


export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# missing entirely before - job ran under the bare system python (no cv2), see run_robosuite_eval.sh
# in this same directory for the matching pattern.
export PATH=/mnt/beegfs/frosa/.conda/envs/tinyvla_robosuite_1_0_1_provola/bin:$PATH
PKL_PATH=/mnt/beegfs/frosa/checkpoint_save_folder/checkpoint_save_folder/tiny_vla/post_processed_tiny_vla_llava_pythia_lora_ur5e_pick_place_rm_12_13_14_15_lora_r_128_processed/checkpoint-40000/rollout_pick_place_0_False_obj_set_-1_change_command_False_use_vllm_False_otd_False
srun python create_video.py \
    --path_to_pkl $PKL_PATH \
    --output_dir $PKL_PATH/video \
    --only_failures