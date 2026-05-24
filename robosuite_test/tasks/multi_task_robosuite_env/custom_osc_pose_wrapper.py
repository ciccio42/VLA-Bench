from robosuite.wrappers.wrapper import Wrapper
import numpy as np
from pyquaternion import Quaternion
import robosuite.utils.transform_utils as T
import copy
import cv2

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
osc_pose_logger = logging.getLogger(name="OSCPOSELogger")

def _as_list(value):
    """Return value as a list, preserving existing lists / tuples."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _first(value):
    """Return first element for robosuite args that may be scalar or list."""
    return _as_list(value)[0]


def _camera_depth_enabled(camera_depths, index=0):
    if isinstance(camera_depths, (list, tuple)):
        return bool(camera_depths[index])
    return bool(camera_depths)


def _eef_site_id(env, arm="right"):
    """robosuite <=1.4 used an int; robosuite >=1.5 may use a per-arm dict."""
    site_id = env.robots[0].eef_site_id
    if isinstance(site_id, dict):
        if arm in site_id:
            return site_id[arm]
        return next(iter(site_id.values()))
    return site_id


def _split_step_result(step_result):
    """Support both robosuite's 4-tuple and Gymnasium-style 5-tuple."""
    if len(step_result) == 4:
        return step_result
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    raise ValueError(f"Unexpected env.step return length: {len(step_result)}")



class CustomOSCPoseWrapper(Wrapper):
    def __init__(self, env, ranges):
        super().__init__(env)
        self.action_repeat = 5
        self.ranges = ranges

    def _get_rel_action(self, action, base_pos, base_quat):
        if action.shape[0] == 7:
            cmd_quat = T.axisangle2quat(action[3:6])
            quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
            aa = T.quat2axisangle(quat)
            return np.concatenate((action[:3] - base_pos, aa, action[6:]))
        else:
            cmd_quat = Quaternion(angle=action[3] * np.pi, axis=action[4:7])
            cmd_quat = np.array(
                [cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
            quat = T.quat_multiply(T.quat_inverse(base_quat), cmd_quat)
            aa = T.quat2axisangle(quat)
            return np.concatenate((action[:3] - base_pos, aa, action[7:]))

    def _project_point(self, point, sim, camera='agentview', frame_width=320, frame_height=320):
        model_matrix = np.zeros((3, 4))
        model_matrix[:3, :3] = sim.data.get_camera_xmat(camera).T

        fovy = sim.model.cam_fovy[sim.model.camera_name2id(camera)]
        f = 0.5 * frame_height / np.tan(fovy * np.pi / 360)
        camera_matrix = np.array(
            ((f, 0, frame_width / 2), (0, f, frame_height / 2), (0, 0, 1)))

        MVP_matrix = camera_matrix.dot(model_matrix)
        cam_coord = np.ones((4, 1))
        cam_coord[:3, 0] = point - sim.data.get_camera_xpos(camera)

        clip = MVP_matrix.dot(cam_coord)
        row, col = clip[:2].reshape(-1) / clip[2]
        row, col = row, frame_height - col
        return int(max(col, 0)), int(max(row, 0))

    def _get_real_depth(self, depth_img):
        # Make sure that depth values are normalized
        assert np.all(depth_img >= 0.0) and np.all(depth_img <= 1.0)
        extent = self.env.sim.model.stat.extent
        far = self.env.sim.model.vis.map.zfar * extent
        near = self.env.sim.model.vis.map.znear * extent
        return near / (1.0 - depth_img * (1.0 - near / far))

    def post_proc_obs(self, obs, env):
        new_obs = {}
        from PIL import Image
        robot_name = env.robots[0].robot_model.naming_prefix
        for k in obs.keys():
            if k.startswith(robot_name):
                name = k[len(robot_name):]
                if isinstance(obs[k], np.ndarray):
                    new_obs[name] = obs[k].copy()
                else:
                    new_obs[name] = obs[k]
            else:
                if isinstance(obs[k], np.ndarray):
                    new_obs[k] = obs[k].copy()
                else:
                    new_obs[k] = obs[k]

        frame_height, frame_width = _first(self.env.camera_heights), _first(self.env.camera_widths)
        if self.env.use_camera_obs:
            for camera_name in self.env.camera_names:
                # save image observation
                new_width = int(obs[f"{camera_name}_image"].shape[1] * 1)
                new_height = int(obs[f"{camera_name}_image"].shape[0] * 1)
                new_dim = (new_width, new_height)
                # cv2.imwrite("pre_flip_debug_img.png",
                #             obs[f"{camera_name}_image"])
                new_obs[f"{camera_name}_image"] = cv2.resize(
                    obs[f"{camera_name}_image"].copy()[::-1,], new_dim)
                # cv2.imwrite("post_flip_debug_img.png",
                #             new_obs[f"{camera_name}_image"])
                if _camera_depth_enabled(self.env.camera_depths):
                    new_obs[f"{camera_name}_depth_norm"] = np.array(
                        obs[f"{camera_name}_depth"].copy()[::-1]*255, dtype=np.uint8)
                    new_obs[f"{camera_name}_depth"] = self._get_real_depth(
                        obs[f"{camera_name}_depth"]).copy()[::-1]

        aa = T.quat2axisangle(obs[robot_name+'eef_quat'])
        flip_points = np.array(self._project_point(obs[robot_name+'eef_pos'], env.sim,
                                                   camera="camera_front", frame_width=frame_width, frame_height=frame_height))
        flip_points[0] = frame_height - flip_points[0]
        flip_points[1] = frame_width - flip_points[1]
        new_obs['extent'] = self.env.sim.model.stat.extent
        new_obs['zfar'] = self.env.sim.model.vis.map.zfar
        new_obs['znear'] = self.env.sim.model.vis.map.znear
        new_obs['eef_point'] = flip_points
        new_obs['ee_aa'] = np.concatenate(
            (obs[robot_name+'eef_pos'], aa)).astype(np.float32)
        return new_obs

    @staticmethod
    def _canonical_quat_xyzw(quat):
        """Return a normalized xyzw quaternion with a stable sign convention."""
        quat = np.asarray(quat, dtype=np.float64).copy()
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        quat /= norm
        # q and -q encode the same pose. Keeping w >= 0 prevents axis-angle
        # discontinuities near pi that can make the wrist flip.
        if quat[3] < 0.0:
            quat *= -1.0
        return quat

    @staticmethod
    def _quat_error_local_xyzw(current_quat, desired_quat):
        """
        Orientation error expected by robosuite OSC_POSE.

        robosuite OSC_POSE position deltas are expressed in the MuJoCo world
        frame, but orientation deltas are relative to the current end-effector
        frame. Therefore, for an absolute desired world orientation q_des and
        current world orientation q_cur, the action rotation is:

            q_err = inverse(q_cur) * q_des

        encoded as a scaled axis-angle vector.
        """
        q_cur = CustomOSCPoseWrapper._canonical_quat_xyzw(current_quat)
        q_des = CustomOSCPoseWrapper._canonical_quat_xyzw(desired_quat)

        # Use the equivalent desired quaternion closest to q_cur, so the
        # relative rotation follows the shortest path.
        if np.dot(q_cur, q_des) < 0.0:
            q_des *= -1.0

        q_err = T.quat_multiply(T.quat_inverse(q_cur), q_des)
        q_err = CustomOSCPoseWrapper._canonical_quat_xyzw(q_err)
        aa = T.quat2axisangle(q_err)

        # Numerical safety: axis-angle has a singularity at pi. This keeps the
        # command just below pi and avoids controller jumps.
        angle = np.linalg.norm(aa)
        if angle > np.pi:
            aa = aa / angle * (angle - 2.0 * np.pi)
        return aa

    def convert_rotation_gripper_to_world(self, action, base_pos, base_quat):
        """
        Convert the expert's absolute world target-pose action into robosuite's
        OSC_POSE action convention.

        Input action convention preserved from the old code:
            [target_x, target_y, target_z, abs_axis_angle_x, abs_axis_angle_y,
             abs_axis_angle_z, gripper]

        robosuite OSC_POSE convention:
            [world_delta_x, world_delta_y, world_delta_z, local_delta_axis_angle_x,
             local_delta_axis_angle_y, local_delta_axis_angle_z, gripper]
        """
        action = np.asarray(action, dtype=np.float64)

        if action.shape[0] == 7:
            desired_quat = T.axisangle2quat(action[3:6])
            ori_delta = self._quat_error_local_xyzw(base_quat, desired_quat)
            return np.concatenate((action[:3] - base_pos, ori_delta, action[6:]))

        # Legacy alternate format: [pos(3), angle, axis(3), gripper...]
        cmd_quat = Quaternion(angle=action[3], axis=action[4:7])
        desired_quat = np.array([cmd_quat.x, cmd_quat.y, cmd_quat.z, cmd_quat.w])
        ori_delta = self._quat_error_local_xyzw(base_quat, desired_quat)
        return np.concatenate((action[:3] - base_pos, ori_delta, action[7:]))

    def step(self, action):
        reward = -100.0
        osc_pose_logger.debug("-------------------------------------------")
        #for _ in range(self.action_repeat):
        dist = np.inf
        cnt = 10
        prev_dist = 0.0
        #
        #while dist > 0.004:
        #for _ in range(self.action_repeat):
        while dist > 0.004 and cnt > 0:
            # take the current position and gripper orientation with respect to world
            osc_pose_logger.debug(f"Target position {action[:3]}")
            base_pos = self.env.sim.data.site_xpos[_eef_site_id(self.env)]
            base_quat = T.mat2quat(np.reshape(
                self.env.sim.data.site_xmat[_eef_site_id(self.env)], (3, 3)))
            global_action = self.convert_rotation_gripper_to_world(
                action, base_pos, base_quat)
            osc_pose_logger.debug(f"Global delta position {global_action[:3]}")
            obs, reward_t, done, info = _split_step_result(self.env.step(global_action))
            reward = max(reward, reward_t)
            
            dist = np.linalg.norm(
                self.env.sim.data.site_xpos[_eef_site_id(self.env)] - action[:3])
            
            if round(prev_dist,4) == round(dist,4):
                # if the distance does not change, the robot is not moving
                # print(f"Robot is not moving, {dist}")
                cnt -= 1
            else:
                prev_dist = dist
            
        osc_pose_logger.debug(
            "----------------------------------------------\n\n")
        return self.post_proc_obs(obs, self.env), reward, done, info

    def reset(self, *args, **kwargs):
        reset_result = super().reset(*args, **kwargs)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            return self.post_proc_obs(obs, self.env), info
        return self.post_proc_obs(reset_result, self.env)

    def _get_observation(self):
        if hasattr(self.env, "_get_observation"):
            obs = self.env._get_observation()
        else:
            obs = self.env._get_obs()
        return self.post_proc_obs(obs, self.env)
