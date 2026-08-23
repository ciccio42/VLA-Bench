import base64
import time

import cv2
import numpy as np
import requests
from robosuite.utils.transform_utils import mat2euler, quat2mat

from .configs import LeRobotPolicyConfig

# Both constants copied from openvla_utils.py verbatim, not reimported (that module has an
# unconditional `import tensorflow`, unavailable in this conda env — see euler_to_axis_angle
# below for the same reason). Both are properties of this benchmark's dataset/env, not of any
# specific model family, so they apply here exactly as they do for OpenVLA/TinyVLA:
#
# SCALE_FACTOR: the dataset's raw `action` field (and therefore what any model trained on it
# predicts) is stored pre-scaled by 1/SCALE_FACTOR relative to real meters/radians — confirmed
# empirically: raw action position-delta magnitudes are ~0.1-0.3 (implausibly large for a single
# control step), and dataset gripper values range 0-20 (pick_place.py's gripper thresholds of
# 0.75/0.5 only make sense post-scaling, i.e. against a ~0-1 range). Skipping this step is what
# caused the arm to fly off the table in early testing: unscaled deltas of "0.1-0.3 per axis" were
# being added directly onto eef_pos every single step.
SCALE_FACTOR = 0.05

# R_EE_TO_GRIPPER: a fixed axis remapping (not a live rotation) between robosuite's raw
# `eef_quat` frame and the "gripper" frame the dataset's orientation state/deltas are actually
# expressed in. Used both when building the state fed to the policy (must match training) and
# when accumulating the predicted orientation delta (must match how the delta was computed at
# collection time). Omitting this was the second cause of drift.
R_EE_TO_GRIPPER = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

# Policy types whose own postprocessor already outputs a directly-usable gripper command
# (e.g. VLA-JEPA's binarize_gripper_action=True gives {-1, +1} regardless of the dataset's raw
# 0-20 scale) — SCALE_FACTOR must NOT be applied to the gripper dim for these, or a "close"
# prediction of +1 becomes +0.05, which pick_place.py's >0.75 threshold can never register.
GRIPPER_ALREADY_SCALED_POLICY_TYPES = {"vla_jepa"}


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gripper_frame_euler(quat):
    return np.array([normalize_angle(a) for a in mat2euler(R_EE_TO_GRIPPER @ quat2mat(quat))])


def euler_to_axis_angle(euler):
    # Inlined from openvla_utils.euler_to_axis_angle (same math, so both model families
    # accumulate orientation the same way) rather than imported: that module does an
    # unconditional `import tensorflow`, which isn't installed in this (tinyvla) conda env.
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx

    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0):
        return np.zeros(3)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis * angle


class lerobot_remote_policy:
    def __init__(self, cfg: LeRobotPolicyConfig):
        self.cfg = cfg
        self.url = f"http://127.0.0.1:{cfg.server_port}/predict"
        self.chunk_size = cfg.chunk_size
        # fail fast if the server isn't up yet, rather than timing out on the first real request
        requests.get(f"http://127.0.0.1:{cfg.server_port}/", timeout=10)

    def _encode_image(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok, "failed to encode image as PNG"
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _build_state(self, obs, gripper_closed):
        joint_pos = np.asarray(obs["joint_pos"], dtype=np.float32)[:6]
        eef_pos = np.asarray(obs["eef_pos"], dtype=np.float32)
        eef_euler = gripper_frame_euler(obs["eef_quat"]).astype(np.float32)
        # 13-D, matches convert_dataset.py::make_frame's observation.state layout:
        # [joint_0..5, gripper_closed, eef_x,y,z,roll,pitch,yaw]
        state = np.concatenate([joint_pos, [float(gripper_closed)], eef_pos, eef_euler])
        return state.tolist()

    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name="pick_place", n_steps=-1):
        start = time.time()
        images = {
            "front": obs["camera_front_image"],
            "gripper": obs["robot0_eye_in_hand_image"],
        }
        payload = {
            "images": {k: self._encode_image(v) for k, v in images.items()},
            "state": self._build_state(obs, gripper_closed),
            "task_description": task_description,
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        resp_json = resp.json()
        delta = np.array(resp_json["action"], dtype=np.float64)  # dx,dy,dz,droll,dpitch,dyaw,gripper
        policy_type = resp_json.get("policy_type")
        # The model predicts position/orientation in the dataset's pre-scaled convention (see
        # SCALE_FACTOR above) — matches openvla.py::action_post_processing. Confirmed via the
        # VLA-JEPA smoke test that its gripper dim must NOT get this same scaling (see
        # GRIPPER_ALREADY_SCALED_POLICY_TYPES above): a +1 "close" prediction became +0.05 and
        # was silently read as "open" by pick_place.py's thresholds.
        delta[0:6] = delta[0:6] * SCALE_FACTOR
        if policy_type not in GRIPPER_ALREADY_SCALED_POLICY_TYPES:
            delta[6] = delta[6] * SCALE_FACTOR
        elapsed = time.time() - start

        action_world = np.zeros(7)
        action_world[0:3] = obs["eef_pos"] + delta[0:3]
        current_euler = gripper_frame_euler(obs["eef_quat"])
        target_euler = [normalize_angle(a) for a in (current_euler + delta[3:6])]
        action_world[3:6] = euler_to_axis_angle(target_euler)
        action_world[6] = delta[6]

        # Single-action chunk: re-query the server every env step, exactly like lerobot-eval's own
        # rollout() does. Simpler and safer than replicating LeRobot's internal action-queue
        # chunking on the client side.
        return [action_world], elapsed
