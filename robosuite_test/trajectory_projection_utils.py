"""Shared helpers for test_trajectory_projection.py.

Three independent pieces live here:
  1. A dependency-free loader for the raw rollout .pkl files written by
     multi_task_il.datasets.savers.Trajectory (see Multi-Task-LFD-Training-Framework/
     training/multi_task_il/datasets/savers.py). We deliberately do NOT import that
     module directly: its package import chain pulls in mujoco_py/robosuite, which
     aren't installed in the openvla-oft/tinyvla conda envs this script actually runs
     under. Since the pickled class only needs to survive attribute access (get/
     __getitem__/__len__), a structural stand-in plus a pickle.Unpickler that remaps
     the class reference is sufficient and has zero heavy dependencies.
  2. Simulated-camera projection (world frame -> pixel), replicating
     robosuite.utils.camera_utils's exact intrinsic/extrinsic formulas so the
     projection matches how the training images were actually rendered.
  3. Real-camera projection (robot base_link frame -> pixel), replicating the
     transform chain in UR5e-2f-85/ai_controller/ai_controller/script_controller/
     vision.py (camera <-> ArUco <-> table_0) plus the table_0 <-> base_link static
     transform documented in UR5e-2f-85/docs/script_controller.md, run in reverse
     (that code deprojects pixels to base_link; we need the inverse, base_link points
     to pixels). Real ZED intrinsics are not checked into the repo anywhere (they are
     normally read live from a ROS CameraInfo topic) -- see APPROXIMATE_REAL_INTRINSICS
     below.
"""
from __future__ import annotations

import pickle
from typing import Optional

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# 1. Raw rollout .pkl loading
# ---------------------------------------------------------------------------


def _decompress_obs(obs: dict) -> dict:
    """Verbatim port of multi_task_il.datasets.savers._decompress_obs."""
    keys = ["camera_front_image", "eye_in_hand_image"]
    for key in keys:
        if key in obs and "image" in key:
            try:
                obs[key] = cv2.imdecode(obs[key], cv2.IMREAD_COLOR)
            except Exception:
                pass
        if key in obs and "depth_norm" in key:
            obs[key] = cv2.imdecode(obs[key], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return obs


class _MinimalTrajectory:
    """Structural stand-in for multi_task_il.datasets.savers.Trajectory.

    Pickle reconstructs this via __new__ + direct __dict__ update (the real
    Trajectory defines no __getstate__/__reduce__), so no __init__ is needed --
    only the read accessors used by this script.
    """

    def get(self, t, decompress=True):
        obs_t, reward_t, done_t, info_t, action_t = self._data[t]
        if decompress:
            obs_t = _decompress_obs(dict(obs_t))
        return dict(obs=obs_t, reward=reward_t, done=done_t, info=info_t, action=action_t)

    def __getitem__(self, t):
        return self.get(t)

    def __len__(self):
        return len(self._data)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Trajectory" and module.endswith("savers"):
            return _MinimalTrajectory
        return super().find_class(module, name)


def load_rollout_pkl(path: str) -> dict:
    """Load a raw rollout .pkl, returning the pickled payload dict (has a 'traj' key
    holding a _MinimalTrajectory, plus metadata like 'len'/'env_type'/'task_id')."""
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# 2. Simulated-camera projection (robosuite / MuJoCo convention)
# ---------------------------------------------------------------------------

# MuJoCo's camera body frame has +Z pointing away from the view direction and a
# flipped Y; this correction is exactly robosuite.utils.camera_utils.get_camera_extrinsic_matrix's
# camera_axis_correction, needed to get a standard OpenCV camera frame (X right, Y down, Z forward).
_SIM_CAMERA_AXIS_CORRECTION = np.diag([1.0, -1.0, -1.0, 1.0])


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _make_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def _pose_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def load_sim_camera_pose(camera_config_path: str, camera_name: str = "camera_front"):
    """Read a camera's world-frame pose + FOV out of a robosuite task yaml
    (e.g. PickPlaceDistractor.yaml)."""
    with open(camera_config_path) as f:
        cfg = yaml.safe_load(f)
    pos, quat_wxyz = cfg["camera_poses"][camera_name]
    fovy_deg = float(cfg["camera_attribs"]["fovy"])
    return np.asarray(pos, dtype=np.float64), np.asarray(quat_wxyz, dtype=np.float64), fovy_deg


def project_world_points_sim(
    points_world: np.ndarray,
    camera_pos: np.ndarray,
    camera_quat_wxyz: np.ndarray,
    fovy_deg: float,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    """Project Nx3 world-frame points to Nx2 (u, v) pixel coordinates, using the same
    formulas as robosuite.utils.camera_utils.get_camera_{intrinsic,extrinsic}_matrix.
    image_height/image_width should be the ACTUAL image being drawn on (not necessarily
    the yaml's configured camera_heights/camera_widths, in case of a later resize) --
    the focal length is re-derived from fovy + the actual height so it stays correct
    under any uniform resize.
    """
    rot = _quat_wxyz_to_matrix(camera_quat_wxyz)
    camera_pose = _make_pose(camera_pos, rot) @ _SIM_CAMERA_AXIS_CORRECTION
    world_to_camera = _pose_inv(camera_pose)

    f = 0.5 * image_height / np.tan(fovy_deg * np.pi / 360.0)
    K = np.array([[f, 0, image_width / 2.0], [0, f, image_height / 2.0], [0, 0, 1]])
    K_exp = np.eye(4)
    K_exp[:3, :3] = K
    transform = K_exp @ world_to_camera

    points = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    projected = (transform @ points_h.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels


# ---------------------------------------------------------------------------
# 3. Real-camera projection (ZED + ArUco + table_0 + base_link chain)
# ---------------------------------------------------------------------------

# Fixed extrinsic offset between the raw ArUco marker frame and the table_0 TF frame:
# zero translation, 180 degree rotation about Z. Verbatim from
# ai_controller/ai_controller/script_controller/vision.py's ARUCO_TO_TABLE0_ROTATION
# (self-inverse: this matrix is both its own transpose and its own inverse).
_ARUCO_TO_TABLE0_ROTATION = np.diag([-1.0, -1.0, 1.0])

# table_0 <-> base_link static transform, from the `static_transform_publisher` command
# documented in UR5e-2f-85/docs/script_controller.md (--frame-id base_link --child-frame-id
# table_0): translation (x, y, z) and quaternion (0, 0, 1, 0) = 180 degrees about Z, which
# gives the same diag(-1, -1, 1) rotation matrix as the ArUco leg above (also self-inverse).
# This is a manually-measured constant, not computed anywhere in code.
TABLE0_TO_BASE_TRANSLATION = np.array([0.00, 0.612, -0.120])
TABLE0_TO_BASE_ROTATION = np.diag([-1.0, -1.0, 1.0])

# No ZED intrinsics are saved anywhere in the repo (get_camera_intrinsics() in vision.py
# fetches them live from a ROS CameraInfo topic at eval time). This is a best-effort
# placeholder derived only from the ZED Mini's published ~90 degree horizontal FOV
# (Stereolabs spec, not from this repo) at whatever resolution the loaded image actually
# is -- NOT calibrated per-unit. Override with real values (capture one CameraInfo
# message: `ros2 topic echo /zed_front/zed_node/rgb/color/rect/camera_info -f`, its `k`
# field is the row-major 3x3 K) via --real_camera_intrinsics fx,fy,cx,cy as soon as they're
# available; treat anything drawn using this fallback as approximate.
_APPROXIMATE_REAL_HFOV_DEG = 90.0


def load_real_camera_calibration(calibration_path: str, camera_key: str = "zed_front"):
    with open(calibration_path) as f:
        raw = yaml.safe_load(f)
    entry = raw[camera_key]
    return np.asarray(entry["position"], dtype=np.float64), np.asarray(
        entry["orientation_matrix"], dtype=np.float64
    )


def approximate_real_intrinsics(image_width: int, image_height: int, hfov_deg: float = _APPROXIMATE_REAL_HFOV_DEG) -> np.ndarray:
    fx = (image_width / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    fy = fx
    return np.array([[fx, 0, image_width / 2.0], [0, fy, image_height / 2.0], [0, 0, 1]])


def project_base_points_real(
    points_base: np.ndarray,
    camera_position_aruco: np.ndarray,
    camera_orientation_matrix_aruco: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project Nx3 points expressed in the robot's base_link frame to Nx2 (u, v) pixel
    coordinates, by running vision.py's pixel->base_link chain in reverse:
    base_link -> table_0 -> ArUco -> camera optical frame -> pixel.
    """
    points = np.atleast_2d(np.asarray(points_base, dtype=np.float64))

    # base_link -> table_0 (inverse of table_0 -> base_link; rotation is self-inverse)
    points_table0 = (TABLE0_TO_BASE_ROTATION @ (points - TABLE0_TO_BASE_TRANSLATION).T).T
    # table_0 -> ArUco (inverse of ArUco -> table_0; also self-inverse)
    points_aruco = (_ARUCO_TO_TABLE0_ROTATION @ points_table0.T).T
    # ArUco -> camera optical frame (inverse of vision.py's camera_point_to_aruco:
    # point_aruco = R_cm @ point_cam + t_cm)
    points_cam = ((camera_orientation_matrix_aruco.T) @ (points_aruco - camera_position_aruco).T).T

    # Standard pinhole forward projection (inverse of vision.py's deproject_pixel).
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * points_cam[:, 0] / points_cam[:, 2] + cx
    v = fy * points_cam[:, 1] / points_cam[:, 2] + cy
    return np.stack([u, v], axis=1)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_projected_trajectories(
    image: np.ndarray,
    start_px: Optional[np.ndarray],
    gt_px: np.ndarray,
    pred_px: np.ndarray,
) -> np.ndarray:
    """Draw the start point, GT trajectory (green), and predicted trajectory (red) onto
    a copy of `image` (BGR uint8), with a small legend. Points are (u, v) float pairs."""
    out = image.copy()
    h, w = out.shape[:2]

    def _clip(pt):
        return int(np.clip(pt[0], 0, w - 1)), int(np.clip(pt[1], 0, h - 1))

    def _draw_series(points, color, label_prefix):
        prev = None
        for i, pt in enumerate(points):
            xy = _clip(pt)
            cv2.circle(out, xy, 4, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(out, str(i + 1), (xy[0] + 5, xy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            if prev is not None:
                cv2.line(out, prev, xy, color, 1, lineType=cv2.LINE_AA)
            prev = xy

    if start_px is not None:
        xy = _clip(start_px)
        cv2.drawMarker(out, xy, (255, 128, 0), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2)

    _draw_series(gt_px, (0, 200, 0), "GT")
    _draw_series(pred_px, (0, 0, 255), "Pred")

    legend_y = 18
    for text, color in [("start", (255, 128, 0)), ("ground truth", (0, 200, 0)), ("predicted", (0, 0, 255))]:
        cv2.putText(out, text, (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        legend_y += 18

    return out
