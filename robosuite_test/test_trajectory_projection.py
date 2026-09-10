"""Sanity-check a trained OpenVLA-OFT or TinyVLA checkpoint by: loading one raw
training rollout .pkl, running the model from its first frame, and drawing both the
ground-truth trajectory (from the recorded rollout) and the model's predicted
trajectory (from its output action chunk) as projected 2D points on the starting
camera_front frame.

Reuses the exact same policy wrapper classes (robosuite_test/models/{openvla,tinyvla}.py)
and config schema (robosuite_test/models/configs.py) as run_robosuite_eval.py, so any
existing eval config YAML (e.g. models/openvla_eval_config.yml) works unchanged --
just append --pkl_path/--dataset_type/--task_description on top.

Usage (run from this directory, robosuite_test/, same convention as run_robosuite_eval.py):
    python test_trajectory_projection.py \
        --config_path=models/openvla_eval_config.yml \
        --pkl_path=/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/pick_place/real_new_ur5e_pick_place/task_00/traj000.pkl \
        --dataset_type=real \
        --task_description="Pick the green box and place it into the first bin"

    python test_trajectory_projection.py \
        --config_path=models/tinyvla_eval_config.yml \
        --pkl_path=/mnt/beegfs/frosa/robot_datasets/dataset/no_opt_dataset/pick_place/ur5e_pick_place/task_09/traj058.pkl \
        --dataset_type=sim \
        --task_description="Pick the green box and place it into the first bin"

For the real-world case, camera intrinsics are not available anywhere in the repo
(see trajectory_projection_utils.py's APPROXIMATE_REAL_INTRINSICS docstring) -- this
script prints a loud warning and falls back to an approximation unless
--real_camera_intrinsics fx,fy,cx,cy is given.
"""
import copy
import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

sys.path.append("../.")
sys.path.append("./robosuite/robosuite")

import draccus

from models.configs import EvalConfig
from trajectory_projection_utils import (
    approximate_real_intrinsics,
    draw_projected_trajectories,
    load_real_camera_calibration,
    load_rollout_pkl,
    load_sim_camera_pose,
    project_base_points_real,
    project_world_points_sim,
)

DEFAULT_SIM_CAMERA_CONFIG = (
    "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/"
    "tasks/multi_task_robosuite_env/config/PickPlaceDistractor.yaml"
)
DEFAULT_REAL_CAMERA_CALIBRATION = "/mnt/beegfs/frosa/Multi-Task-LFD-Framework/repo/estimated_camera_positions.yaml"


@dataclass
class TrajectoryProjectionConfig(EvalConfig):
    pkl_path: str = ""  # raw rollout .pkl (Trajectory-pickled, NOT an eval-output rollout)
    dataset_type: str = "real"  # "sim" or "real" -- picks the camera projection math
    task_description: str = "Pick the object and place it into the bin"
    frame_index: int = 0  # start inference from this frame of the pkl
    sim_camera_config_path: str = DEFAULT_SIM_CAMERA_CONFIG
    real_camera_calibration_path: str = DEFAULT_REAL_CAMERA_CALIBRATION
    real_camera_intrinsics: Optional[str] = None  # "fx,fy,cx,cy" override for the real dataset
    output_path: str = "./trajectory_projection_outputs"


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # Matches ur5e_pick_place.py's `traj[t]['obs']['camera_front_image'][:, :, ::-1]` --
    # the raw pkl's images are decoded via cv2 (BGR); the TFDS dataset (and therefore what
    # every model was actually trained on) is RGB.
    return image[:, :, ::-1]


def load_policy(cfg: TrajectoryProjectionConfig):
    if cfg.model_family.lower() == "openvla":
        from models.openvla import open_vla_policy

        return open_vla_policy(cfg.model_config)
    elif cfg.model_family.lower() == "tinyvla":
        from models.tinyvla import llava_pythia_act_policy

        return llava_pythia_act_policy(cfg.model_config)
    raise ValueError(f"Unsupported model_family: {cfg.model_family!r} (expected 'openvla' or 'tinyvla')")


def get_resize_size(cfg: TrajectoryProjectionConfig):
    from robot_utils import get_image_resize_size

    return get_image_resize_size(cfg)


def project_points(cfg: TrajectoryProjectionConfig, points_3d: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    if cfg.dataset_type == "sim":
        camera_pos, camera_quat_wxyz, fovy_deg = load_sim_camera_pose(cfg.sim_camera_config_path, "camera_front")
        return project_world_points_sim(points_3d, camera_pos, camera_quat_wxyz, fovy_deg, image_height, image_width)
    elif cfg.dataset_type == "real":
        camera_pos_aruco, camera_rot_aruco = load_real_camera_calibration(cfg.real_camera_calibration_path, "zed_front")
        if cfg.real_camera_intrinsics:
            fx, fy, cx, cy = (float(v) for v in cfg.real_camera_intrinsics.split(","))
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            print(
                "WARNING: no --real_camera_intrinsics given and none are checked into the repo "
                "(ZED intrinsics are normally read live from a ROS CameraInfo topic) -- falling "
                "back to an approximation from the ZED Mini's published ~90deg HFOV. Projected "
                "pixels for the real dataset are APPROXIMATE; capture a real CameraInfo message "
                "and pass --real_camera_intrinsics fx,fy,cx,cy for accurate overlays."
            )
            intrinsics = approximate_real_intrinsics(image_width, image_height)
        return project_base_points_real(points_3d, camera_pos_aruco, camera_rot_aruco, intrinsics)
    raise ValueError(f"Unsupported dataset_type: {cfg.dataset_type!r} (expected 'sim' or 'real')")


@draccus.wrap()
def main(cfg: TrajectoryProjectionConfig):
    assert cfg.pkl_path, "--pkl_path is required"
    print(f"Loading rollout: {cfg.pkl_path}")
    payload = load_rollout_pkl(cfg.pkl_path)
    traj = payload["traj"]
    episode_len = len(traj)
    print(f"Episode length: {episode_len} steps")

    frame0 = traj[cfg.frame_index]["obs"]
    # Keep an untouched RGB copy of the starting frame for drawing -- compute_action()
    # crops/resizes obs['camera_front_image'] in place as a side effect.
    start_image_rgb = _bgr_to_rgb(frame0["camera_front_image"]).copy()
    start_eef_pos = np.asarray(frame0["eef_pos"], dtype=np.float64)

    policy = load_policy(cfg)
    resize_size = get_resize_size(cfg)

    obs_for_policy = copy.deepcopy(frame0)
    obs_for_policy["camera_front_image"] = _bgr_to_rgb(obs_for_policy["camera_front_image"])
    obs_for_policy["eye_in_hand_image"] = _bgr_to_rgb(obs_for_policy["eye_in_hand_image"])
    gripper_closed = int(round(float(np.asarray(frame0.get("gripper_qpos", 0)).reshape(-1)[0])))

    print("Running inference from frame", cfg.frame_index, "...")
    predicted_actions, elapsed = policy.compute_action(
        obs=obs_for_policy,
        resize_size=resize_size,
        gripper_closed=gripper_closed,
        task_description=cfg.task_description,
        task_name="pick_place",
        n_steps=0,
    )
    print(f"Inference took {elapsed:.2f}s, predicted {len(predicted_actions)} chunk steps")

    pred_positions = np.stack([np.asarray(a[:3], dtype=np.float64) for a in predicted_actions], axis=0)

    horizon = min(len(predicted_actions), episode_len - 1 - cfg.frame_index)
    gt_positions = np.stack(
        [np.asarray(traj[cfg.frame_index + i]["obs"]["eef_pos"], dtype=np.float64) for i in range(1, horizon + 1)],
        axis=0,
    )
    pred_positions = pred_positions[:horizon]

    if horizon <= 0:
        print("WARNING: not enough remaining frames in this rollout to compare against a ground-truth trajectory.")

    h, w = start_image_rgb.shape[:2]
    start_px = project_points(cfg, start_eef_pos[None, :], h, w)[0]
    gt_px = project_points(cfg, gt_positions, h, w) if horizon > 0 else np.zeros((0, 2))
    pred_px = project_points(cfg, pred_positions, h, w) if horizon > 0 else np.zeros((0, 2))

    annotated = draw_projected_trajectories(cv2.cvtColor(start_image_rgb, cv2.COLOR_RGB2BGR), start_px, gt_px, pred_px)

    os.makedirs(cfg.output_path, exist_ok=True)
    pkl_stem = os.path.splitext(os.path.basename(cfg.pkl_path))[0]
    out_file = os.path.join(cfg.output_path, f"{cfg.model_family}_{cfg.dataset_type}_{pkl_stem}_frame{cfg.frame_index}.png")
    cv2.imwrite(out_file, annotated)
    print(f"Saved annotated frame to {out_file}")

    if horizon > 0:
        errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
        print("Per-step 3D position error (m):", np.round(errors, 4).tolist())
        print(f"Mean 3D position error over {horizon} steps: {errors.mean():.4f} m")


if __name__ == "__main__":
    main()
