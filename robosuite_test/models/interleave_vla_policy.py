import base64
import io
import os
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from robosuite.utils.transform_utils import mat2euler, quat2mat

from .configs import InterleaveVLAConfig
from robosuite_utils import ENV_OBJECTS, TASK_CROP

# All three constants below are properties of this benchmark's UR5e pick_place dataset/action
# convention, not of any specific model family -- copied verbatim from lerobot_policy.py (see
# that file's comments for the empirical evidence: unscaled actions flew the arm off the table,
# and the frame mismatch caused orientation drift).
SCALE_FACTOR = 0.05
R_EE_TO_GRIPPER = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

# Native robosuite render resolution for the pick_place task family (TASK_MAP['pick_place']
# ['render_hw'] in robosuite_utils.py) -- matches the resolution the real/sim TFDS datasets'
# `bounding_boxes` were computed at (real_ur5e_pick_place.py's observation_spec camera_front_image
# is also (200, 360, 3)). Bounding-box projection below is done at THIS resolution, then run
# through the exact same crop-then-resize transform (TASK_CROP + resize to 224) that
# ur5e_pick_place.py / ur5e_interleave_grounding_bin.py apply, so the crops fed to the model at
# inference match the distribution it was trained on.
RENDER_HEIGHT, RENDER_WIDTH = 200, 360
INSTRUCTION_IMAGE_SIZE = 224

TARGET_TO_BBOX = {
    "green box": "greenbox",
    "yellow box": "yellowbox",
    "blue box": "bluebox",
    "red box": "redbox",
}
BIN_ORDINAL_TO_INDEX = {"first": 0, "second": 1, "third": 2, "fourth": 3}
BIN_HIGHLIGHT_COLOR = (180, 0, 255)
BIN_PADDING_COLOR = (184, 167, 72)


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gripper_frame_euler(quat):
    return np.array([normalize_angle(a) for a in mat2euler(R_EE_TO_GRIPPER @ quat2mat(quat))])


def euler_to_axis_angle(euler):
    # Inlined rather than imported from openvla_utils (unconditional `import tensorflow`,
    # unavailable in this conda env) -- identical math to lerobot_policy.py's copy, so all
    # three model families accumulate orientation the same way.
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


def _project_box_to_pixel_bbox(center, half_dims, camera_pos, camera_quat_wxyz, fovy_deg):
    """Project an axis-aligned 3D box's 8 corners to native-render pixel space, return
    (x_min, y_min, x_max, y_max) at RENDER_WIDTH x RENDER_HEIGHT resolution."""
    from trajectory_projection_utils import project_world_points_sim

    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners = center[None, :] + signs * half_dims[None, :]
    pixels = project_world_points_sim(
        corners, camera_pos, camera_quat_wxyz, fovy_deg, RENDER_HEIGHT, RENDER_WIDTH
    )
    x_min, y_min = pixels[:, 0].min(), pixels[:, 1].min()
    x_max, y_max = pixels[:, 0].max(), pixels[:, 1].max()
    return x_min, y_min, x_max, y_max


def _crop_params():
    top, bottom_margin, left, right_margin = TASK_CROP["pick_place"]
    box_h = RENDER_HEIGHT - top - bottom_margin
    box_w = RENDER_WIDTH - left - right_margin
    return top, left, box_h, box_w


def _remap_bbox_to_cropped_resized(bbox, top, left, box_h, box_w):
    """Maps a (x_min, y_min, x_max, y_max) bbox in native-render pixel space into the
    TASK_CROP-cropped-then-resized-to-INSTRUCTION_IMAGE_SIZE frame -- same transform as
    ur5e_pick_place.py::crop_image_adj_bb / ur5e_interleave_grounding_bin.py's builders."""
    scale_x = INSTRUCTION_IMAGE_SIZE / box_w
    scale_y = INSTRUCTION_IMAGE_SIZE / box_h
    x_min, y_min, x_max, y_max = bbox
    x_min = np.clip((x_min - left) * scale_x, 0, INSTRUCTION_IMAGE_SIZE - 1)
    x_max = np.clip((x_max - left) * scale_x, 0, INSTRUCTION_IMAGE_SIZE - 1)
    y_min = np.clip((y_min - top) * scale_y, 0, INSTRUCTION_IMAGE_SIZE - 1)
    y_max = np.clip((y_max - top) * scale_y, 0, INSTRUCTION_IMAGE_SIZE - 1)
    return x_min, y_min, x_max, y_max


class interleave_vla_remote_policy:
    def __init__(self, cfg: InterleaveVLAConfig):
        self.cfg = cfg
        self.url = f"http://127.0.0.1:{cfg.server_port}/predict"
        self.chunk_size = cfg.chunk_size
        # fail fast if the server isn't up yet, rather than timing out on the first real request
        requests.get(f"http://127.0.0.1:{cfg.server_port}/", timeout=10)

        from trajectory_projection_utils import load_sim_camera_pose
        self._camera_pos, self._camera_quat_wxyz, self._fovy_deg = load_sim_camera_pose(
            cfg.sim_camera_config_path, "camera_front"
        )

        self._target_crop_b64 = None
        self._bin_grounding_b64 = None
        self._table_perpendicular_roll_pitch = None

        self.debug_save_images = getattr(cfg, "debug_save_images", False)
        self.debug_image_dir = getattr(cfg, "debug_image_dir", "./debug_interleave_images")
        if self.debug_save_images:
            os.makedirs(self.debug_image_dir, exist_ok=True)
        self._debug_episode_idx = -1
        self._debug_step_idx = 0

    def reset(self):
        # New episode: the target-object crop and bin-grounding image are derived from the
        # FIRST frame only (matches the TFDS builders, which build them once per episode from
        # `steps[0]`), so drop any cached ones from the previous episode.
        self._target_crop_b64 = None
        self._bin_grounding_b64 = None
        self._table_perpendicular_roll_pitch = None
        self._debug_episode_idx += 1
        self._debug_step_idx = 0

    def _encode_image(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok, "failed to encode image as PNG"
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _decode_image_b64(self, b64_png: str) -> np.ndarray:
        return np.array(Image.open(io.BytesIO(base64.b64decode(b64_png))).convert("RGB"))

    def _save_debug_image(
        self, front_image: np.ndarray, command_text: str, target_crop_b64: str, bin_grounding_b64: str
    ):
        """Saves the exact preprocessed 'front' image sent to the model this step (with the
        command overlaid), to debug_image_dir -- for visually checking the crop/resize
        pipeline against training-time examples. target_crop/bin_grounding are cached per
        episode (built once from the first frame, see _build_instruction_images), so they're
        only written once per episode (at step 0) rather than duplicated every step."""
        img = Image.fromarray(front_image).convert("RGB")
        draw = ImageDraw.Draw(img)
        margin = 4
        text_w, text_h = draw.textbbox((0, 0), command_text)[2:]
        draw.rectangle(
            [0, img.height - text_h - 2 * margin, img.width, img.height],
            fill=(0, 0, 0),
        )
        draw.text((margin, img.height - text_h - margin), command_text, fill=(255, 255, 0))
        ep = self._debug_episode_idx
        filename = f"ep{ep:03d}_step{self._debug_step_idx:04d}_front.png"
        img.save(os.path.join(self.debug_image_dir, filename))

        if self._debug_step_idx == 0:
            target_crop = self._decode_image_b64(target_crop_b64)
            bin_grounding = self._decode_image_b64(bin_grounding_b64)
            Image.fromarray(target_crop).save(os.path.join(self.debug_image_dir, f"ep{ep:03d}_target_crop.png"))
            Image.fromarray(bin_grounding).save(os.path.join(self.debug_image_dir, f"ep{ep:03d}_bin_grounding.png"))

    def _build_instruction_images(self, obs, task_description, task_name):
        """Builds (and caches) the target-object crop + bin-grounding image from the
        episode's first frame, exactly mirroring
        tfds_dataset_builder/ur5e_(sim_)interleave/ur5e_interleave_grounding_bin.py's
        `_extract_target_crop` / `_extract_bin_grounding_image` -- but from ground-truth sim
        object positions (projected via trajectory_projection_utils) instead of pre-computed
        `bounding_boxes`, since this benchmark's env doesn't expose those directly."""
        if self._target_crop_b64 is not None:
            return self._target_crop_b64, self._bin_grounding_b64

        objs = ENV_OBJECTS[task_name]
        top, left, box_h, box_w = _crop_params()
        front_image = obs["camera_front_image"]
        # Native render resolution assumed to match RENDER_HEIGHT/RENDER_WIDTH -- if the env
        # was configured at a different camera_heights/camera_widths, this crop math (and the
        # bbox projection above) silently misaligns; see this module's RENDER_HEIGHT docstring.
        assert front_image.shape[:2] == (RENDER_HEIGHT, RENDER_WIDTH), (
            f"Expected native render {(RENDER_HEIGHT, RENDER_WIDTH)}, got {front_image.shape[:2]}"
        )

        # --- target object crop ---
        target_name = None
        for phrase, bbox_key in TARGET_TO_BBOX.items():
            if phrase in task_description.lower():
                target_name = bbox_key
                break
        if target_name is None:
            raise ValueError(f"Cannot identify target object in instruction: {task_description!r}")

        target_center = np.asarray(obs[f"{target_name}_pos"], dtype=np.float64)
        target_half_dims = np.asarray(objs["obj_dim"][target_name], dtype=np.float64) / 2.0
        bbox = _project_box_to_pixel_bbox(
            target_center, target_half_dims, self._camera_pos, self._camera_quat_wxyz, self._fovy_deg
        )
        # Expand 20% each side, matching the TFDS builders' `_extract_target_crop`.
        x_min, y_min, x_max, y_max = bbox
        w, h = x_max - x_min, y_max - y_min
        bbox = (x_min - 0.2 * w, y_min - 0.2 * h, x_max + 0.2 * w, y_max + 0.2 * h)
        x_min, y_min, x_max, y_max = _remap_bbox_to_cropped_resized(bbox, top, left, box_h, box_w)

        cropped_resized = cv2.resize(
            front_image[top:top + box_h, left:left + box_w],
            (INSTRUCTION_IMAGE_SIZE, INSTRUCTION_IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        xi_min, yi_min, xi_max, yi_max = int(x_min), int(y_min), int(max(x_min + 1, x_max)), int(max(y_min + 1, y_max))
        target_crop = cropped_resized[yi_min:yi_max, xi_min:xi_max]
        if target_crop.size == 0:
            target_crop = cropped_resized
        target_crop = cv2.resize(target_crop, (INSTRUCTION_IMAGE_SIZE, INSTRUCTION_IMAGE_SIZE))

        # --- bin grounding image ---
        bin_ordinal = None
        for ordinal in BIN_ORDINAL_TO_INDEX:
            if f"{ordinal} bin" in task_description.lower():
                bin_ordinal = ordinal
                break
        if bin_ordinal is None:
            raise ValueError(f"Cannot identify target bin in instruction: {task_description!r}")
        target_bin_index = BIN_ORDINAL_TO_INDEX[bin_ordinal]

        bin_x, _, bin_z = objs["bin_position"]
        # obj_dim entries are documented as [W, H, D] (robosuite_utils.py comment). For the
        # single "bin" body, W=0.6 is the FULL tray's length along world Y (spanning all 4
        # compartments -- NOT a per-compartment extent), H=0.06 is the tray's vertical wall
        # height (world Z), D=0.15 is the tray's front-back depth (world X). Each sub-bin's own
        # box uses D/2 for its X half-extent and H/2 for its Z half-extent; the Y half-extent
        # comes from the per-compartment `ranges` width instead of W (see sub_bin_half_width_y).
        bin_w, bin_h, bin_d = np.asarray(objs["obj_dim"]["bin"], dtype=np.float64)
        bin_y_ranges = objs["ranges"]  # left-to-right, matches single_bin_0..3 training convention
        sub_bin_half_width_y = abs(bin_y_ranges[0][1] - bin_y_ranges[0][0]) / 2.0

        bin_pixel_boxes = []
        for y_lo, y_hi in bin_y_ranges:
            center = np.array([bin_x, (y_lo + y_hi) / 2.0, bin_z])
            half_dims = np.array([bin_d / 2.0, sub_bin_half_width_y, bin_h / 2.0])
            bin_pixel_boxes.append(
                _project_box_to_pixel_bbox(center, half_dims, self._camera_pos, self._camera_quat_wxyz, self._fovy_deg)
            )

        group_x_min = min(b[0] for b in bin_pixel_boxes)
        group_y_min = min(b[1] for b in bin_pixel_boxes)
        group_x_max = max(b[2] for b in bin_pixel_boxes)
        group_y_max = max(b[3] for b in bin_pixel_boxes)
        margin_x = (group_x_max - group_x_min) * 0.10
        margin_y = (group_y_max - group_y_min) * 0.10
        crop_x_min = max(0, group_x_min - margin_x)
        crop_y_min = max(0, group_y_min - margin_y)
        crop_x_max = min(RENDER_WIDTH, group_x_max + margin_x)
        crop_y_max = min(RENDER_HEIGHT, group_y_max + margin_y)

        bin_group_crop = front_image[int(crop_y_min):int(crop_y_max), int(crop_x_min):int(crop_x_max)]
        if bin_group_crop.size == 0:
            bin_group_crop = front_image
        orig_h, orig_w = bin_group_crop.shape[:2]
        scale = min(INSTRUCTION_IMAGE_SIZE / orig_w, INSTRUCTION_IMAGE_SIZE / orig_h)
        resized_w, resized_h = max(1, int(round(orig_w * scale))), max(1, int(round(orig_h * scale)))
        resized_bin_crop = cv2.resize(bin_group_crop, (resized_w, resized_h))

        canvas = np.full((INSTRUCTION_IMAGE_SIZE, INSTRUCTION_IMAGE_SIZE, 3), BIN_PADDING_COLOR, dtype=np.uint8)
        pad_left = (INSTRUCTION_IMAGE_SIZE - resized_w) // 2
        pad_top = INSTRUCTION_IMAGE_SIZE - resized_h  # bins occupy the lower band, matches training
        canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized_bin_crop

        target_box = bin_pixel_boxes[target_bin_index]
        tx_min = int(np.clip(pad_left + (target_box[0] - crop_x_min) * scale, 0, INSTRUCTION_IMAGE_SIZE - 1))
        ty_min = int(np.clip(pad_top + (target_box[1] - crop_y_min) * scale, 0, INSTRUCTION_IMAGE_SIZE - 1))
        tx_max = int(np.clip(pad_left + (target_box[2] - crop_x_min) * scale, 0, INSTRUCTION_IMAGE_SIZE - 1))
        ty_max = int(np.clip(pad_top + (target_box[3] - crop_y_min) * scale, 0, INSTRUCTION_IMAGE_SIZE - 1))
        cx = (tx_min + tx_max) // 2
        cy = (ty_min + ty_max) // 2

        bin_img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(bin_img)
        draw.rectangle([(tx_min, ty_min), (tx_max, ty_max)], outline=BIN_HIGHLIGHT_COLOR, width=2)
        draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], fill=BIN_HIGHLIGHT_COLOR)
        bin_grounding = np.asarray(bin_img)

        self._target_crop_b64 = self._encode_image(target_crop)
        self._bin_grounding_b64 = self._encode_image(bin_grounding)
        return self._target_crop_b64, self._bin_grounding_b64

    def _build_state(self, obs, gripper_closed):
        eef_pos = np.asarray(obs["eef_pos"], dtype=np.float32)
        eef_euler = gripper_frame_euler(obs["eef_quat"]).astype(np.float32)
        # 7-D, matches ur5e_interleave_grounding_bin.py's state layout:
        # [x, y, z, roll, pitch, yaw, gripper]
        return np.concatenate([eef_pos, eef_euler, [float(gripper_closed)]]).tolist()

    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name="pick_place", n_steps=-1):
        start = time.time()
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        target_crop_b64, bin_grounding_b64 = self._build_instruction_images(obs, task_description, task_name)

        # Crop then resize, matching openvla.py::compute_action and the TASK_CROP-cropped
        # training distribution (ur5e_pick_place.py / ur5e_interleave_grounding_bin.py) --
        # resizing the raw uncropped render would feed the model a different field of view
        # than it was trained on.
        image = obs["camera_front_image"]
        crop_top, crop_bottom, crop_left, crop_right = TASK_CROP[task_name]
        img_height, img_width = image.shape[0], image.shape[1]
        box_h = img_height - crop_top - crop_bottom
        box_w = img_width - crop_left - crop_right
        cropped_image = image[crop_top:crop_top + box_h, crop_left:crop_left + box_w]

        # Same two-stage resize as openvla.py::compute_action/prepare_observation: an
        # intermediate cv2 resize to 224x224, then a JPEG encode/decode roundtrip + a
        # lanczos3+antialias resize to resize_size -- this replicates the JPEG compression
        # and resize_image() (src/data/dlimp/utils.py) that the TFDS builder applies when
        # writing images (encoding_format='jpeg'), which raw/uncompressed cv2 output alone
        # doesn't have. TensorFlow isn't available in this conda env, so PIL stands in for
        # both the JPEG roundtrip and the lanczos resize (Image.LANCZOS is numerically very
        # close to tf.image.resize's lanczos3).
        front_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        jpeg_buf = io.BytesIO()
        Image.fromarray(front_image).save(jpeg_buf, format="JPEG", quality=95)
        jpeg_buf.seek(0)
        front_image = np.array(Image.open(jpeg_buf).convert("RGB"))
        front_image = np.array(
            Image.fromarray(front_image).resize(resize_size[::-1], Image.LANCZOS)
        )

        interleaved_instruction = task_description
        for phrase in TARGET_TO_BBOX:
            if phrase in interleaved_instruction.lower():
                idx = interleaved_instruction.lower().index(phrase)
                interleaved_instruction = interleaved_instruction[:idx] + "<image>" + interleaved_instruction[idx + len(phrase):]
                break
        for ordinal in BIN_ORDINAL_TO_INDEX:
            phrase = f"{ordinal} bin"
            if phrase in interleaved_instruction.lower():
                idx = interleaved_instruction.lower().index(phrase)
                interleaved_instruction = interleaved_instruction[:idx] + "<image>" + interleaved_instruction[idx + len(phrase):]
                break

        if self.debug_save_images:
            self._save_debug_image(front_image, interleaved_instruction, target_crop_b64, bin_grounding_b64)
            self._debug_step_idx += 1

        payload = {
            "images": {
                "front": self._encode_image(front_image),
                "target_crop": target_crop_b64,
                "bin_grounding": bin_grounding_b64,
            },
            "state": self._build_state(obs, gripper_closed),
            "language_instruction": interleaved_instruction,
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        delta = np.array(resp.json()["action"], dtype=np.float64)  # dx,dy,dz,droll,dpitch,dyaw,gripper
        delta = delta * SCALE_FACTOR  # same raw-dataset pre-scaling as lerobot_policy.py/openvla.py

        action_world = np.zeros(7)
        action_world[0:3] = obs["eef_pos"] + delta[0:3]
        current_euler = gripper_frame_euler(obs["eef_quat"])
        target_euler = [normalize_angle(a) for a in (current_euler + delta[3:6])]

        # Lock roll/pitch to the episode's initial (top-down) pose, same reasoning/heuristic as
        # lerobot_policy.py -- see that file's comment for why unlocked droll/dpitch are noise,
        # not a deliberate correction, for this dataset.
        if n_steps == 0 or self._table_perpendicular_roll_pitch is None:
            self._table_perpendicular_roll_pitch = (current_euler[0], current_euler[1])
        target_euler[0], target_euler[1] = self._table_perpendicular_roll_pitch

        action_world[3:6] = euler_to_axis_angle(target_euler)
        action_world[6] = delta[6]

        elapsed = time.time() - start
        # Single-action chunk: re-query the server every env step, same choice as lerobot_policy.py.
        return [action_world], elapsed
