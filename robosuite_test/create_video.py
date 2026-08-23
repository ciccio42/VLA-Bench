import argparse
import glob
import json
import pickle as pkl
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import imageio
import debugpy
import sys


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
        first_img = traj[0]['obs']['camera_front_image']
        height, width, _ = first_img.shape

        video_name = os.path.join(args.output_dir, os.path.basename(pkl_file).replace('.pkl', '.mp4'))

        with imageio.get_writer(video_name, fps=10, codec='libx264') as writer:
            for t in range(len(traj) - 1):
                img = traj[t]['obs']['camera_front_image']
                img_pil = Image.fromarray(img).convert("RGBA")

                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default(size=14)

                overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                text_bg_padding = 6
                corner_radius = 8
                shadow_offset = (3, 3)
                bottom_margin = 10
                right_margin = 10

                display_text = task_description.replace(" and ", " and\n", 1)

                # Compute text bounding box
                bbox = draw.multiline_textbbox((0, 0), display_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_position = (
                    width - text_width - 2 * text_bg_padding - right_margin,
                    height - text_height - 2 * text_bg_padding - bottom_margin,
                )

                rect_start = (text_position[0] - text_bg_padding, text_position[1] - text_bg_padding)
                rect_end = (text_position[0] + text_width + text_bg_padding, text_position[1] + text_height + text_bg_padding)

                shadow_start = (rect_start[0] + shadow_offset[0], rect_start[1] + shadow_offset[1])
                shadow_end = (rect_end[0] + shadow_offset[0], rect_end[1] + shadow_offset[1])

                # Shadow first, then the rounded box on top, then the text
                draw.rounded_rectangle([shadow_start, shadow_end], radius=corner_radius, fill=(0, 0, 0, 120))
                draw.rounded_rectangle([rect_start, rect_end], radius=corner_radius, fill=(0, 0, 0, 200))
                draw.multiline_text(text_position, display_text, fill=(255, 255, 255, 255), font=font)

                img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")

                writer.append_data(np.array(img_pil))

        print(f"Saved video to {video_name}")
