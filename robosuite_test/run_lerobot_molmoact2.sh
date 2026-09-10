#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --export=ALL

sbatch run_lerobot_eval.sh \
        models/lerobot_molmoact2_eval_config.yml \
        /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/lerobot/lerobot/outputs/ur5e_molmoact2_bsz16/checkpoints/032000/pretrained_model\
        8765 \
        0 \
        10