import sys
from pathlib import Path

if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
import numpy as np
from multi_task_il.datasets import Trajectory
try:
    import pybullet as p
except:
    pass
from pyquaternion import Quaternion
import random
from robosuite.utils.transform_utils import quat2axisangle
try:
    from robosuite.utils import RandomizationError
except Exception:
    try:
        from robosuite.utils.errors import RandomizationError
    except Exception:
        class RandomizationError(Exception):
            pass
import os
# import mujoco_py
import robosuite.utils.transform_utils as T
from multi_task_robosuite_env import get_env

try:
    # Prefer the robosuite-1.5 compatible wrapper if it is next to this file.
    from custom_osc_pose_wrapper_rs15 import CustomOSCPoseWrapper
except Exception:
    try:
        from custom_osc_pose_wrapper import CustomOSCPoseWrapper
    except Exception:
        CustomOSCPoseWrapper = None
import cv2
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
pick_place_logger = logging.getLogger(name="PickPlaceLogger")

object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

# in case rebuild is needed to use GPU render: sudo mkdir -p /usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# pip uninstall mujoco_py; pip install mujoco_py




def _canonical_quat_xyzw(quat, reference=None):
    """Normalize xyzw quaternion and choose a stable sign.

    If reference is provided, choose the equivalent sign closest to reference.
    This prevents axis-angle / SLERP discontinuities that can flip the wrist.
    """
    quat = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        quat /= norm
    if reference is not None:
        ref = _canonical_quat_xyzw(reference)
        if np.dot(quat, ref) < 0.0:
            quat *= -1.0
    elif quat[3] < 0.0:
        quat *= -1.0
    return quat


def _canonical_pyquat(quat, reference=None):
    q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)
    q = _canonical_quat_xyzw(q, reference=reference)
    return Quaternion(q[3], q[0], q[1], q[2])

def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step

def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _eef_site_id(env, arm="right"):
    """Return an integer MuJoCo site id for the robot end-effector.

    robosuite <=1.4 often exposed robot.eef_site_id as an int.
    robosuite 1.5 may expose it as a dict, e.g. {"right": id}.
    NumPy arrays such as sim.data.site_xmat require an integer index.
    """
    site_id = env.robots[0].eef_site_id

    if isinstance(site_id, dict):
        if arm in site_id:
            site_id = site_id[arm]
        else:
            site_id = next(iter(site_id.values()))

    return int(site_id)


def _unwrap_env(env):
    """Return the underlying robosuite env if a Wrapper is used."""
    return getattr(env, "env", env)


def _is_absolute_pose_wrapper(env):
    return hasattr(env, "post_proc_obs") and hasattr(env, "action_repeat")


def _ensure_absolute_pose_wrapper(env, ranges):
    """
    The expert policy emits absolute world target actions:
      [target_x, target_y, target_z, target_axis_angle(3), gripper]
    Native robosuite OSC_POSE consumes controller-space actions, so we keep
    the old action contract by wrapping the env if get_env() did not already do it.
    """
    if _is_absolute_pose_wrapper(env):
        return env
    if CustomOSCPoseWrapper is None:
        raise RuntimeError(
            "CustomOSCPoseWrapper is required to keep the existing absolute-pose action interface, "
            "but it could not be imported."
        )
    return CustomOSCPoseWrapper(env, ranges)


def _split_step_result(step_result):
    if len(step_result) == 4:
        return step_result
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    raise ValueError(f"Unexpected env.step return length: {len(step_result)}")


def _split_reset_result(reset_result):
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0]
    return reset_result


def _get_flattened_sim_state(sim):
    """
    robosuite historically exposed sim.get_state().flatten().
    Keep that path, with a MuJoCo-data fallback for newer simulators.
    """
    if hasattr(sim, "get_state"):
        state = sim.get_state()
        return state.flatten() if hasattr(state, "flatten") else np.array(state).ravel()

    data = sim.data
    chunks = [np.asarray(data.qpos).ravel(), np.asarray(data.qvel).ravel()]
    if hasattr(data, "act") and data.act is not None:
        chunks.append(np.asarray(data.act).ravel())
    return np.concatenate(chunks)


def _set_flattened_sim_state(sim, flat_state):
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(flat_state)
        return

    # Fallback for newer MuJoCo-style bindings.
    flat_state = np.asarray(flat_state)
    nq = sim.model.nq
    nv = sim.model.nv
    sim.data.qpos[:] = flat_state[:nq]
    sim.data.qvel[:] = flat_state[nq:nq + nv]
    if hasattr(sim.data, "act") and sim.data.act is not None:
        na = len(sim.data.act)
        if flat_state.shape[0] >= nq + nv + na:
            sim.data.act[:] = flat_state[nq + nv:nq + nv + na]


def _forward_sim(sim):
    if hasattr(sim, "forward"):
        sim.forward()
    elif hasattr(sim, "mj_forward"):
        sim.mj_forward()


def _load_controller_config(ctrl_config, robot_type="UR5e", arms=("right",)):
    """
    Accepts:
      - a robosuite 1.5 composite dict,
      - a robosuite <=1.4 part-controller dict,
      - a part controller name such as OSC_POSE / IK_POSE,
      - a composite controller name such as BASIC,
      - a path to either a composite JSON or old part-controller JSON.
    Returns a robosuite 1.5-compatible composite controller config whenever possible.
    """
    import importlib
    import os

    if isinstance(ctrl_config, dict):
        # If it is already composite, keep it. If it is an old part config, refactor it.
        if "body_parts" in ctrl_config:
            return ctrl_config
        if ctrl_config.get("type") in {"IK_POSE", "OSC_POSE", "OSC_POSITION", "JOINT_POSITION", "JOINT_VELOCITY", "JOINT_TORQUE"}:
            try:
                from robosuite.controllers.composite.composite_controller_factory import (
                    refactor_composite_controller_config,
                )
                return refactor_composite_controller_config(
                    ctrl_config,
                    robot_type=robot_type,
                    arms=list(arms),
                )
            except Exception:
                return ctrl_config
        return ctrl_config

    controllers = importlib.import_module("robosuite.controllers")
    load_composite = getattr(controllers, "load_composite_controller_config", None)
    load_part = getattr(controllers, "load_part_controller_config", None)
    load_legacy = getattr(controllers, "load_controller_config", None)

    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except Exception:
        refactor_composite_controller_config = None

    part_controller_names = {
        "IK_POSE",
        "OSC_POSE",
        "OSC_POSITION",
        "JOINT_POSITION",
        "JOINT_VELOCITY",
        "JOINT_TORQUE",
    }

    if isinstance(ctrl_config, str) and ctrl_config in part_controller_names:
        if load_part is None:
            if load_legacy is not None:
                return load_legacy(default_controller=ctrl_config)
            raise RuntimeError(f"No part controller loader found for {ctrl_config}")

        part_config = load_part(default_controller=ctrl_config)
        if refactor_composite_controller_config is not None:
            return refactor_composite_controller_config(
                part_config,
                robot_type=robot_type,
                arms=list(arms),
            )
        return part_config

    if isinstance(ctrl_config, str) and os.path.isfile(ctrl_config):
        if load_composite is not None:
            try:
                return load_composite(controller=ctrl_config)
            except Exception:
                pass

        if load_part is not None:
            part_config = load_part(custom_fpath=ctrl_config)
            if refactor_composite_controller_config is not None:
                return refactor_composite_controller_config(
                    part_config,
                    robot_type=robot_type,
                    arms=list(arms),
                )
            return part_config

        if load_legacy is not None:
            return load_legacy(custom_fpath=ctrl_config)

        raise RuntimeError(f"No controller loader found for file: {ctrl_config}")

    # Composite controller default, e.g. BASIC.
    if load_composite is not None:
        return load_composite(controller=ctrl_config)

    if load_legacy is not None:
        return load_legacy(default_controller=ctrl_config)

    raise RuntimeError(f"No controller loader found for controller: {ctrl_config}")


class PickPlaceController:
    def __init__(self, env, ranges, tries=0, object_set=1):
        self._env = env
        self.ranges = ranges
        self._object_set = object_set
        self.reset()

    def _calculate_quat(self, obs):
        obj_quat = obs["{}_quat".format(self._object_name)]

        if "UR5e" in self._env.robot_names:
            # Keep Robotiq2f-85 orientation stable.
            # This preserves the current downward grasp convention.
            return self._base_quat

        if "nut" in self._object_name:
            obj_rot = T.quat2mat(obj_quat)
            obj_rot = np.array(
                [[-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0]]
            ) @ obj_rot
        else:
            obj_rot = T.quat2mat(obj_quat)

        world_ee_rot = np.matmul(obj_rot, self._target_gripper_wrt_obj_rot)
        return Quaternion(matrix=world_ee_rot)

    def reset(self):
        self._object_name = self._env.objects[self._env.object_id].name
        # TODO this line violates abstraction barriers but so does the reference implementation in robosuite
        self._jpos_getter = lambda: np.array(self._env._joint_positions)
        self._clearance = 0.03  # 0.03 if 'milk' not in self._object_name else -0.01

        if "Sawyer" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 1e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
            self._g_tol = 1e-2
        elif "Panda" in self._env.robot_names:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.02
            self._final_thresh = 6e-2
            # define the target gripper orientation with respect to the object
            self._target_gripper_wrt_obj_rot = np.array(
                [[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._g_tol = 5e-2
        elif "UR5e" in self._env.robot_names:
            self._obs_name = "eef_pos"
            self._default_speed = 0.02
            self._final_thresh = 6e-2
            self._g_tol = 5e-2

            # Robotiq2f-85 / UR5e: keep the initial gripper orientation as the
            # default grasp orientation instead of forcing the legacy target matrix.
            eef_site_id = _eef_site_id(self._env)
            current_eef_rot = np.reshape(
                self._env.sim.data.site_xmat[eef_site_id],
                (3, 3),
            )

            # For boxes, this prevents the gripper from flipping upside down.
            self._target_gripper_wrt_obj_rot = current_eef_rot.copy()
        else:
            raise NotImplementedError

        # define the initial orientation of the gripper site
        self._base_quat = Quaternion(matrix=np.reshape(
            self._env.sim.data.site_xmat[_eef_site_id(self._env)], (3, 3)))
        pick_place_logger.debug(
            f"Starting position:\n{self._env.sim.data.site_xpos[_eef_site_id(self._env)]}")
        pick_place_logger.debug(
            f"Base rot:\n{np.reshape(self._env.sim.data.site_xmat[_eef_site_id(self._env)], (3,3))}")

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.20
        self._obj_thr = 0.10
        if self._object_set == 1:
            dist_panda = {'milk': 0.05, 'can': 0.018,
                          'cereal': 0.018, 'bread': 0.018}
            dist_sawyer = {'milk': 0.05, 'can': 0.018,
                           'cereal': 0.018, 'bread': 0.018}
            dist_ur5e = {'milk': 0.05, 'can': 0.03,
                         'cereal': 0.03, 'bread': 0.03}
            self.final_placing = [0, 0, 0.12]
        elif self._object_set == 2:
            dist_panda = {'greenbox': 0.05, 'yellowbox': 0.018,
                          'bluebox': 0.018, 'redbox': 0.018}
            dist_sawyer = {'greenbox': 0.05, 'yellowbox': 0.018,
                           'bluebox': 0.018, 'redbox': 0.018}
            dist_ur5e = {'greenbox': 0.05, 'yellowbox': 0.03,
                         'bluebox': 0.03, 'redbox': 0.03}
            self.final_placing = [0, 0, 0.12]
        elif self._object_set == 3:
            # {'greenbox': 0.05, 'yellowbox': 0.018,
            #  'bluebox': 0.018, 'redbox': 0.018}
            dist_panda = dict()
            # {'greenbox': 0.05, 'yellowbox': 0.018,
            #  'bluebox': 0.018, 'redbox': 0.018}
            dist_sawyer = dict()
            # {'greenbox': 0.05, 'yellowbox': 0.03,
            #  'bluebox': 0.03, 'redbox': 0.03}
            dist_ur5e = dict()
            for obj_name in self._env.obj_names:
                dist_panda[obj_name] = 0.03
                dist_sawyer[obj_name] = 0.03
                dist_ur5e[obj_name] = 0.03
            self.final_placing = [0, 0, 0.12]

        if "Panda" in self._env.robot_names:
            self.dist = dist_panda
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.01
        elif "Sawyer" in self._env.robot_names:
            self.dist = dist_sawyer
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2
        elif "UR5e" in self._env.robot_names:
            self.dist = dist_ur5e
            # gripper depth defines the distance between the TCP and the edge of the gripper
            self._gripper_depth = 0.038/2

    def _object_in_hand(self, obs):
        # if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name]) < self.dist[self._object_name] \
        #    and (obs['{}_pos'.format(self._object_name)][2] - obs[self._obs_name][2]) > 0 \
        #    and (obs['{}_pos'.format(self._object_name)][2] - obs[self._obs_name][2]) <= self._gripper_depth:
        if np.linalg.norm(obs['{}_pos'.format(self._object_name)] - obs[self._obs_name]) < self.dist[self._object_name]:
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)
        quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)
        quat = _canonical_quat_xyzw(quat)
        aa = quat2axisangle(quat)

        # absolute in world frame
        return np.concatenate((delta_pos + base_pos, aa))

    def act(self, obs):
        self._target_loc = np.array(
            self._env.sim.data.body_xpos[self._env.bin_bodyid]) + [0, 0, self._hover_delta]
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False
            self._target_quat = self._calculate_quat(obs)
            # q and -q represent the same orientation. Align target sign with
            # the starting gripper quaternion so SLERP follows the short path.
            self._target_quat = _canonical_pyquat(
                self._target_quat,
                reference=np.array([self._base_quat.x, self._base_quat.y, self._base_quat.z, self._base_quat.w]),
            )
            self._move_up = False

        # Phase 1
        if self._start_grasp < 0:
            # check if the "prepare_grasp" phase is over
            if np.linalg.norm(obs['{}_pos'.format(self._object_name)][:2] - obs[self._obs_name][:2]) < self._g_tol:
                self._start_grasp = self._t

            # perform the inteporpolation between _base_quat and _target_quat
            quat_t = Quaternion.slerp(
                self._base_quat, self._target_quat, min(1, float(self._t) / 20))
            eef_pose = self._get_target_pose(
                obs['{}_pos'.format(self._object_name)] -
                obs[self._obs_name] + [0, 0, self._hover_delta],
                obs['eef_pos'], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'
        # Phase 2
        elif self._start_grasp > 0 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                # the object is not in the hand, approaching the object
                eef_pose = self._get_target_pose(
                    obs['{}_pos'.format(self._object_name)] -
                    obs[self._obs_name] - [0, 0, self._clearance],
                    obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['{}_pos'.format(self._object_name)]
                status = 'reaching_obj'
            else:
                # the object is in the hand, close the gripper and start the new phase
                eef_pose = self._get_target_pose(
                    self.object_pos - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                self._finish_grasp = True
                status = 'obj_in_hand'
        # Phase 3
        elif np.linalg.norm(
                self._target_loc - obs[self._obs_name]) > self._final_thresh and not self._intermediate_reached:
            if not self._move_up:
                self._init_obj_pos = obs['{}_pos'.format(self._object_name)]
                self._move_up = True
            # check the current object height
            if (abs(self._init_obj_pos[2] - obs['{}_pos'.format(self._object_name)][2]) < self._obj_thr):
                # target location is the current gripper position + security threshold
                target = obs['eef_pos'] + [0, 0, self._obj_thr]
            else:
                target = self._target_loc
            # moving towards the goal bin
            eef_pose = self._get_target_pose(
                target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        # Phase 4
        else:
            self._intermediate_reached = True
            if np.linalg.norm(self._target_loc - self.final_placing - obs[self._obs_name]) > self._final_thresh:
                target = self._target_loc - self.final_placing
                eef_pose = self._get_target_pose(
                    target - obs[self._obs_name], obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
            else:
                eef_pose = self._get_target_pose(
                    np.zeros(3), obs['eef_pos'], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
            status = 'placing'

        self._t += 1
        pick_place_logger.debug(f"Status {status}")
        return action, status


def get_expert_trajectory(env_type, controller_type, renderer=False, camera_obs=True, task=None, ret_env=False, seed=None, env_seed=None, gpu_id=0, render_camera="frontview", object_set=1, **kwargs):
    # assert 'gpu' in str(
    #     mujoco_py.cymj), 'Make sure to render with GPU to make eval faster'
    # Keep render GPU id within visible devices. Do not remap 0->3 etc.
    visible_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_ids:
        n_visible = len([x for x in visible_ids.split(',') if x.strip() != ''])
        if n_visible > 0:
            gpu_id = int(gpu_id) % n_visible
    else:
        gpu_id = max(0, int(gpu_id))
    print(f"GPU-ID {gpu_id}")
    seed = seed if seed is not None else random.getrandbits(32)
    env_seed = seed if env_seed is None else env_seed
    seed_offset = sum([int(a) for a in bytes(env_type, 'ascii')])
    np.random.seed(env_seed)
    if 'Sawyer' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'UR5e' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [-5, 5], [-5, 5], [-5, 5]])
    elif 'Panda' in env_type:
        action_ranges = np.array(
            [[-0.05, 0.25], [-0.45, 0.5], [0.82, 1.2], [0.85, 1.08], [-1, 1], [-1, 1], [-1, 1]])
    if 'Sawyer' in env_type:
        robot_type = "Sawyer"
    elif 'Panda' in env_type:
        robot_type = "Panda"
    elif 'UR5e' in env_type:
        robot_type = "UR5e"
    else:
        robot_type = "UR5e"

    controller_type = _load_controller_config(
        controller_type,
        robot_type=robot_type,
        arms=("right",),
    )

    success, use_object = False, None
    if task is not None:
        assert 0 <= task <= 15, "task should be in [0, 15]"
    else:
        raise NotImplementedError

    if ret_env:
        while True:
            try:
                env = get_env(env_type,
                              controller_configs=controller_type,
                              task_id=task,
                              has_renderer=renderer,
                              has_offscreen_renderer=camera_obs,
                              reward_shaping=False,
                              use_camera_obs=camera_obs,
                              ranges=action_ranges,
                              render_gpu_device_id=gpu_id,
                              render_camera=render_camera,
                              object_set=object_set,
                              ** kwargs)
                env = _ensure_absolute_pose_wrapper(env, action_ranges)
                break
            except RandomizationError:
                pass
        return env

    tries = 0
    while True:
        try:
            env = get_env(env_type,
                          controller_configs=controller_type,
                          task_id=task,
                          has_renderer=renderer,
                          has_offscreen_renderer=camera_obs,
                          reward_shaping=False,
                          use_camera_obs=camera_obs,
                          ranges=action_ranges,
                          render_gpu_device_id=gpu_id,
                          render_camera=render_camera,
                          object_set=object_set,
                          **kwargs)
            env = _ensure_absolute_pose_wrapper(env, action_ranges)
            break
        except RandomizationError:
            pass
    while not success:
        base_env = _unwrap_env(env)
        controller = PickPlaceController(
            base_env, tries=tries, ranges=action_ranges, object_set=object_set)
        np.random.seed(seed + int(tries) + seed_offset)
        while True:
            try:
                obs = _split_reset_result(env.reset())
                break
            except RandomizationError:
                pass
        mj_state = _get_flattened_sim_state(env.sim)
        sim_xml = env.model.get_xml()
        traj = Trajectory(sim_xml)

        env.reset_from_xml_string(sim_xml)
        env.sim.reset()
        _set_flattened_sim_state(env.sim, mj_state)
        _forward_sim(env.sim)
        use_object = base_env.object_id
        traj.append(obs, raw_state=mj_state, info={'status': 'start'})
        print(f"Target object {controller._object_name}")
        for t in range(int(base_env.horizon / env.action_repeat)):
            # compute the action for the current state
            action, status = controller.act(obs)

            obs, reward, done, info = _split_step_result(env.step(action))

            cv2.imwrite(f"debug.png", obs['camera_front_image'][:, :, ::-1])
            try:
                os.makedirs("test")
            except:
                pass
            image = np.array(obs['camera_front_image'][:, :, ::-1])
            for obj_name in obs['obj_bb']['camera_front']:
                obj_bb = obs['obj_bb']['camera_front'][obj_name]
                color = (0, 0, 255)
                image = cv2.rectangle(
                    image, (obj_bb['bottom_right_corner'][0], obj_bb['bottom_right_corner'][1]), (obj_bb['upper_left_corner'][0], obj_bb['upper_left_corner'][1]), color, 1)
            cv2.imwrite(f"test/prova.png",
                        image)
            assert 'status' not in info.keys(
            ), "Don't overwrite information returned from environment. "

            if renderer:
                env.render()

            mj_state = _get_flattened_sim_state(env.sim)
            traj.append(obs, reward, done, info, action, mj_state)
            # # plot bb
            # target_obj_id = obs['target-object']
            # target_obj_bb = None
            # for object_names in object_to_id.keys():
            #     if target_obj_id == object_to_id[object_names]:
            #         target_obj_bb = obs['obj_bb']['camera_front'][object_names]
            # image_rgb = np.array(obs['camera_front_image'][:, :, ::-1])
            # center = target_obj_bb['center']
            # upper_left_corner = target_obj_bb['upper_left_corner']
            # bottom_right_corner = target_obj_bb['bottom_right_corner']
            # image_rgb = cv2.circle(
            #     image_rgb, center, radius=1, color=(0, 0, 255), thickness=-1)
            # image_rgb = cv2.rectangle(
            #     image_rgb, upper_left_corner,
            #     bottom_right_corner, (255, 0, 0), 1)
            # cv2.imshow('camera_front_image', image_rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if reward:
                success = True
                break
        tries += 1

    if renderer:
        env.close()
    del controller
    del env
    return traj


if __name__ == '__main__':
    import debugpy
    import os
    import sys
    import yaml
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    # Load configuration files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    controller_config_path = os.path.join(
        current_dir, "../config/osc_pose.json")
    controller_config = _load_controller_config(
        controller_config_path,
        robot_type="UR5e",
        arms=("right",),
    )

    for i in range(0, 16):
        traj = get_expert_trajectory('UR5e_PickPlaceDistractor',
                                     controller_type=controller_config,
                                     renderer=False,
                                     camera_obs=True,
                                     task=i,
                                     render_camera='camera_front',
                                     object_set=2)
