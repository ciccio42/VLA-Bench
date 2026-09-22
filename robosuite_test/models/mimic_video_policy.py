import base64
import time
from collections import deque

import cv2
import numpy as np
import requests
from robosuite_utils import TASK_CROP

from .configs import MimicVideoConfig

# Final 4:3 canvas this model's training frames were letterboxed to by
# mimic-video/scripts/preprocessing_pipeline.py's resize_with_padding(output_size=VIDEO_SIZE)
# (uniform scale + black-pad, applied directly to the dataset's raw camera frames). Replicated
# here so the eval-time frame goes through the identical operation.
VIDEO_SIZE = (320, 240)

# The world2action action head was trained on video-conditioning embeddings precomputed by
# mimic-video/scripts/run_precompute_video_embeddings.sh with `--obs-history 5 --data-fps 10`:
# every query the action head ever saw during its own training was conditioned on 5 real frames
# spaced at the training data's native 10Hz, never a single static frame (video2world's own
# pretraining randomizes 1-vs-5-frame conditioning, but the action DiT cross-attending into it
# was always trained against the 5-frame flavor -- see world2action_model.py/dataset_video.py).
# This robosuite env's control_freq is 20Hz (mimic_video_policy_server.py's x2 hold-repeat
# comment), so every other captured frame (stride 2) reconstructs that same 10Hz/5-frame window.
IMAGE_HISTORY_STRIDE = 2
IMAGE_HISTORY_LEN = 5


def resize_with_padding(frame: np.ndarray, output_size: tuple = VIDEO_SIZE) -> np.ndarray:
    # Exact port of preprocessing_pipeline.py's resize_with_padding.
    output_width, output_height = output_size
    height, width = frame.shape[:2]
    scale = min(output_width / width, output_height / height)

    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    top = (output_height - resized_height) // 2
    left = (output_width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


class mimic_video_remote_policy:
    # Talks to mimic_video_policy_server.py (mimic-video/model/scripts/) over HTTP -- same
    # bridge pattern as lerobot_policy.py/lerobot_policy_server.py, needed because this model's
    # dependencies (cosmos_predict2/imaginaire/megatron-core, Python 3.10) don't coexist with
    # this harness's own conda env (Python 3.9). See run_mimic_video_eval.sh.
    #
    # Unlike lerobot_remote_policy, all state/rotation-frame conversion and delta-chunk
    # integration happens SERVER-SIDE (mimic_video_policy_server.py): the model predicts a
    # 15-step chunk of deltas relative to a SINGLE observed state, which must be composed
    # sequentially (position addition, rotation matrix composition) rather than applied
    # independently -- that math is model-specific, not shared with other model families, so it
    # doesn't belong duplicated here. This class just does image/state encoding and decodes an
    # already-ready-to-execute world-frame action chunk.
    def __init__(self, cfg: MimicVideoConfig):
        self.cfg = cfg
        self.url = f"http://127.0.0.1:{cfg.server_port}/predict"
        # fail fast if the server isn't up yet, rather than timing out on the first real request
        requests.get(f"http://127.0.0.1:{cfg.server_port}/", timeout=10)
        # Cached by compute_action() for debug_plot_predicted_actions() to default to, so a
        # caller doesn't have to re-thread obs/action_world_chunk through itself just to debug.
        self._last_debug_obs = None
        self._last_debug_action_world_chunk = None
        # Rolling buffer of preprocessed frames, fed by observe() -- see IMAGE_HISTORY_STRIDE's
        # comment. Sized so that after taking every IMAGE_HISTORY_STRIDE-th entry, IMAGE_HISTORY_LEN
        # frames remain (same "(horizon-1)*stride+1" sizing eval/libero/run.py's VAMInference uses).
        self._image_history: deque[np.ndarray] = deque(maxlen=(IMAGE_HISTORY_LEN - 1) * IMAGE_HISTORY_STRIDE + 1)

    def _encode_image(self, img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert ok, "failed to encode image as PNG"
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _preprocess_front_image(self, obs, task_name: str) -> np.ndarray:
        # Sim's raw front-camera render has a wider FOV than the real UR5e camera the underlying
        # open_x_embodiment/ur5e_pick_place_delta_all dataset was collected on -- mimic-video was
        # trained on THAT dataset too (preprocessing_pipeline.py converts it into
        # dataset/ur5e_pick_place_video/), so it needs the same TASK_CROP crop-then-224-resize
        # openvla.py/tinyvla.py/lerobot_policy.py apply, before then going through the
        # resize_with_padding letterbox preprocessing_pipeline.py itself applies to get frames
        # into the fixed 4:3 canvas this model was actually trained at.
        front_image = obs["camera_front_image"]
        crop_params = TASK_CROP[task_name]
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = front_image.shape[0], front_image.shape[1]
        box_h, box_w = img_height - top - crop_params[1], img_width - left - crop_params[3]
        front_image = front_image[top : top + box_h, left : left + box_w]
        front_image = cv2.resize(front_image, (224, 224), interpolation=cv2.INTER_LINEAR)
        return resize_with_padding(front_image)

    def reset(self) -> None:
        """Clear the rolling image-history buffer at the start of a new episode.

        Without this, the buffer (built once in __init__ and shared across every trial in an
        eval run -- see run_robosuite_eval.py's policy construction, which happens before its
        trial loop) keeps whatever frames were left over from the END of the PREVIOUS episode:
        observe()'s own maxlen-padding only fires while the deque is still filling up, so after
        episode 1 it's already full and a single post-reset observe() call only evicts one stale
        frame, leaving up to maxlen-1 of them (a different object pose, possibly a different
        task) in the very first video-conditioning input the model sees for the new episode.
        Matches eval/libero/run.py's VAMInference.reset(), which recreates its own deque the
        same way at every episode boundary.
        """
        self._image_history = deque(maxlen=self._image_history.maxlen)

    def observe(self, obs, task_name: str = "pick_place") -> None:
        """Record one raw env frame into the rolling image-history buffer.

        Unlike compute_action(), which is only invoked once per predicted action chunk (~30 env
        steps apart), the caller (pick_place.py) calls this on EVERY env.step() -- including the
        ones spent replaying an already-fetched chunk -- so that by the time the model is next
        queried, the buffer holds the same kind of real, recently-observed motion the action head
        was trained on (see IMAGE_HISTORY_STRIDE's comment), not just the single instantaneous
        frame at query time.
        """
        processed = self._preprocess_front_image(obs, task_name)
        self._image_history.append(processed)
        while len(self._image_history) < self._image_history.maxlen:
            self._image_history.append(processed.copy())

    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name="pick_place", n_steps=-1):
        start = time.time()
        frames = list(self._image_history)[::IMAGE_HISTORY_STRIDE]

        payload = {
            "images": {"front": [self._encode_image(f) for f in frames]},
            "state": {
                "eef_pos": np.asarray(obs["eef_pos"], dtype=np.float64).tolist(),
                "eef_quat": np.asarray(obs["eef_quat"], dtype=np.float64).tolist(),
                "gripper_closed": float(gripper_closed),
            },
            "task_description": task_description,
        }
        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()
        resp_json = resp.json()
        # Already-composed, ready-to-execute world-frame [x,y,z,axis_angle(3),gripper] entries --
        # pick_place.py's own inner loop steps through all of them (an open-loop chunk) before
        # calling compute_action again. gripper is the raw ~[0,1] predicted value; pick_place.py
        # applies its own >=0.99/<0.5 hysteresis on it, same contract as lerobot_remote_policy.
        action_world_chunk = [np.array(a, dtype=np.float64) for a in resp_json["action_chunk"]]
        elapsed = time.time() - start
        self._last_debug_obs = obs
        self._last_debug_action_world_chunk = action_world_chunk
        return action_world_chunk, elapsed

    @staticmethod
    def _project_point_to_camera_front(point, sim, frame_width, frame_height, camera="camera_front"):
        # Exact port of CustomOSCPoseWrapper._project_point (custom_osc_pose_wrapper.py:33-49)
        # plus the extra flip its only caller applies right after (post_proc_obs, same file,
        # lines 96-99). Kept in lockstep with that implementation rather than re-derived, since
        # it's the one already validated against `camera_front_image` in this exact env (it's
        # what computes `obs['eef_point']`) -- camera_front_image itself is also row-flipped by
        # that same wrapper (`obs[...].copy()[::-1,]`), so any independently-derived projection
        # risks getting that flip's orientation wrong with no easy way to notice.
        model_matrix = np.zeros((3, 4))
        model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

        fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
        f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
        camera_matrix = np.array(((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

        mvp_matrix = camera_matrix.dot(model_matrix)
        cam_coord = np.ones((4, 1))
        cam_coord[:3, 0] = point - sim.data.get_camera_xpos(camera)

        clip = mvp_matrix.dot(cam_coord)
        row, col = clip[:2].reshape(-1) / clip[2]
        row, col = row, frame_height - col
        x, y = int(max(col, 0)), int(max(row, 0))
        # CustomOSCPoseWrapper.post_proc_obs's post-call flip (custom_osc_pose_wrapper.py:98-99),
        # needed to line up with the row-flipped camera_front_image.
        x = frame_height - x
        y = frame_width - y
        # custom_osc_pose_wrapper.py stores this pair as `eef_point` and never draws it as a
        # pixel coordinate directly, so it was never exercised against cv2's (col, row) = (x, y)
        # convention -- traced through the matrix math above, (x, y) here actually work out to
        # (row, col), i.e. swapped. Confirmed empirically too: without this swap the trail
        # clustered in one corner (a 0..frame_height row value used as an x-coordinate only ever
        # lands in the left frame_height/frame_width fraction of a non-square image; a
        # frame_width-scale value used as y overflows a shorter frame_height and mostly clips).
        return y, x

    def debug_plot_predicted_actions(self, env, obs=None, action_world_chunk=None, save_path="debug_predicted_actions.png"):
        """Visual-debug test (not called from the normal eval loop): project the predicted
        world-frame EE positions onto the raw `camera_front_image` and save the annotated frame.

        `env` is the live robosuite env (needed for `env.sim`'s camera pose -- compute_action()
        doesn't have access to it, so this can't run standalone). `obs`/`action_world_chunk`
        default to whatever the most recent compute_action() call cached; pass them explicitly
        to plot a different step. Typical usage:

            action_chunk, _ = policy.compute_action(obs, ...)
            policy.debug_plot_predicted_actions(env)

        Draws on the RAW front-camera frame, not the cropped/letterboxed model-input image --
        projecting onto that would need re-deriving the crop+resize+letterbox chain
        (resize_with_padding above) for the pixel coordinates too, for no benefit here.

        Gripper coloring uses a simple >=0.5 threshold for a quick visual read, not
        pick_place.py's actual >=0.99/<0.5 hysteresis -- this is a debug aid, not a
        control-logic reference.
        """
        obs = obs if obs is not None else self._last_debug_obs
        action_world_chunk = (
            action_world_chunk if action_world_chunk is not None else self._last_debug_action_world_chunk
        )
        assert obs is not None and action_world_chunk is not None, (
            "no obs/action_world_chunk given and none cached -- call compute_action() first, "
            "or pass both explicitly"
        )

        frame_bgr = cv2.cvtColor(obs["camera_front_image"].copy(), cv2.COLOR_RGB2BGR)
        frame_height, frame_width = frame_bgr.shape[:2]

        eef_pos = np.asarray(obs["eef_pos"], dtype=np.float64)
        eef_xy = self._project_point_to_camera_front(eef_pos, env.sim, frame_width, frame_height)
        cv2.circle(frame_bgr, eef_xy, 6, (255, 255, 0), -1)
        cv2.putText(frame_bgr, "current eef", (eef_xy[0] + 8, eef_xy[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

        prev_xy = eef_xy
        for i, action in enumerate(action_world_chunk):
            pos = np.asarray(action[:3], dtype=np.float64)
            gripper = float(action[-1])
            xy = self._project_point_to_camera_front(pos, env.sim, frame_width, frame_height)
            color = (0, 0, 255) if gripper >= 0.5 else (0, 255, 0)  # BGR: red=predicted-closed, green=predicted-open
            cv2.line(frame_bgr, prev_xy, xy, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame_bgr, xy, 4, color, -1)
            cv2.putText(frame_bgr, str(i), (xy[0] + 5, xy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            prev_xy = xy

        cv2.imwrite(save_path, frame_bgr)
        return frame_bgr
