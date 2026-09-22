"""Standalone sanity check for interleave_vla_policy.py's instruction-image construction
(camera projection + crop math), independent of the policy server.

Runs a real PickPlaceDistractor episode reset for a few variations, builds the target-crop +
bin-grounding images the same way interleave_vla_remote_policy.compute_action() would, and
saves them to disk for visual inspection against the training-time examples.

Usage (from this directory, in the tinyvla_robosuite_1_0_1_provola conda env):
    python test_interleave_vla_images.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(".")

from robosuite_utils import build_env_context, startup_env
from models.interleave_vla_policy import interleave_vla_remote_policy
from models.configs import InterleaveVLAConfig

OUT_DIR = "./interleave_vla_image_test"
CONTROLLER_PATH = (
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/VLA-Benchmark/robosuite_test/"
    "tasks/multi_task_robosuite_env/controllers/config/osc_pose.json"
)

VARIATIONS_TO_TEST = [0, 5, 10, 15]  # one per color, spanning all 4 bins


def make_policy_without_server():
    """Builds an interleave_vla_remote_policy instance without requiring a running HTTP
    server -- __init__ normally does a health-check GET, which we don't need just to test
    the image-construction logic."""
    cfg = InterleaveVLAConfig()
    policy = object.__new__(interleave_vla_remote_policy)
    policy.cfg = cfg
    policy.chunk_size = cfg.chunk_size
    from trajectory_projection_utils import load_sim_camera_pose
    policy._camera_pos, policy._camera_quat_wxyz, policy._fovy_deg = load_sim_camera_pose(
        cfg.sim_camera_config_path, "camera_front"
    )
    policy._target_crop_b64 = None
    policy._bin_grounding_b64 = None
    policy._table_perpendicular_roll_pitch = None
    return policy


def decode(b64_png: str) -> np.ndarray:
    import base64
    import io
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("RGB"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import json
    with open("command.json") as f:
        command = json.load(f)

    for variation_id in VARIATIONS_TO_TEST:
        task_description = command["pick_place"][str(variation_id)]
        print(f"=== variation {variation_id}: {task_description!r} ===")

        env = build_env_context(
            env_name="pick_place",
            controller_path=CONTROLLER_PATH,
            variation=variation_id,
            seed=42,
            gpu_id=0,
            object_set=-1,
        )
        done, states, images, obs, traj, tasks, current_gripper_pose = startup_env(
            env=env, variation_id=variation_id, spawn_region=None, num_steps_wait=10,
        )

        policy = make_policy_without_server()
        target_crop_b64, bin_grounding_b64 = policy._build_instruction_images(
            obs, task_description, "pick_place"
        )
        target_crop = decode(target_crop_b64)
        bin_grounding = decode(bin_grounding_b64)

        front = obs["camera_front_image"]
        Image.fromarray(front).save(f"{OUT_DIR}/var{variation_id}_0_front_raw.png")
        Image.fromarray(target_crop).save(f"{OUT_DIR}/var{variation_id}_1_target_crop.png")
        Image.fromarray(bin_grounding).save(f"{OUT_DIR}/var{variation_id}_2_bin_grounding.png")
        print(f"  saved to {OUT_DIR}/var{variation_id}_*.png")

        env.close()

    print(f"\nDone. Inspect images under {OUT_DIR}/")


if __name__ == "__main__":
    main()
