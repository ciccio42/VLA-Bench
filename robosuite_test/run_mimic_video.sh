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
  
