import argparse
import glob
import json
import pickle as pkl
import os
import debugpy
import sys

from robosuite_utils import render_trajectory_video


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_pkl', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--delta_001_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--30000_chkpt/rollout_pick_place")
    parser.add_argument('--output_dir', default="/home/rsofnc000/checkpoint_save_folder/open_vla/openvla-7b+ur5e_pick_place+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--delta_001_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img-gripper_img-proprio--30000_chkpt/rollout_pick_place/videos", help="Directory to save the videos")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for additional output")
    parser.add_argument('--only_failures', action='store_true', help="Only create videos for rollouts whose info_<ctr>.json reports success == 0")
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        

    pkl_files = glob.glob(f"{args.path_to_pkl}/*.pkl")
    pkl_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    os.makedirs(args.output_dir, exist_ok=True)

    for pkl_file in pkl_files:
        if args.only_failures:
            ctr = os.path.basename(pkl_file).split('_')[1].split('.')[0]
            info_file = os.path.join(os.path.dirname(pkl_file), f"info_{ctr}.json")
            if not os.path.exists(info_file):
                print(f"Skipping {pkl_file} (missing {info_file})")
                continue
            with open(info_file, 'r') as f:
                info = json.load(f)
            if info['success']:
                print(f"Skipping {pkl_file} (success={info['success']})")
                continue

        print(f"Loading {pkl_file}")
        with open(pkl_file, 'rb') as f:
            traj = pkl.load(f)

        task_description = traj[0]['obs']['task_description']
        video_name = os.path.join(args.output_dir, os.path.basename(pkl_file).replace('.pkl', '.mp4'))
        render_trajectory_video(traj, task_description, video_name)
        print(f"Saved video to {video_name}")
