#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --exclude=gnode09
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --export=ALL

sbatch run_mimic_video_eval.sh \
        models/mimic_video_eval_config.yml \
        /mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/mimic-video/model/checkpoints/vam/ur5e/w2a_ur5e_pick_place_lr1.000e-04_layer20_bsz16_rotvariance_train \
        8767 \
        0 \
        1
