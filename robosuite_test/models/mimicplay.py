import time
from .configs import MimicPlayConfig
import mimicplay.utils.file_utils as FileUtils
import json
import h5py
from .mimicplay_utils import *
import numpy as np
from PIL import Image

class MimicPlayPolicy:
    def __init__(self, config: MimicPlayConfig):
        # Initialize MimicPlay model with the given configuration
        self.config = config

        # restore policy
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=config.model_path, 
                                                             device=f"cuda", 
                                                             verbose=False,
                                                             train=False,
                                                             highlevel_path=config.highlevel_path)
       
        with open(config.config_path, 'r') as f:
            self.mimicplay_config = json.load(f)
        
        
        self.json_path=self.mimicplay_config['train']['json_path']
        if self.json_path is not None:
            with open(self.json_path, 'r') as f:
                self.agent_demo_id_map = json.load(f)

        self.same_conf = True if 'same_conf' in self.config.model_path else False
        self.same_spawn = True if 'same_spawn' in self.config.model_path else False
        
        
        self.human_dataset_path = self.mimicplay_config['train']['demo_path']
        self.human_demo_dataset = h5py.File(self.human_dataset_path, "r")
        self.valid_demo_ids = list(self.human_demo_dataset['mask']['valid'])        
        
        if not self.same_conf and not self.same_spawn:
            for demo_id in self.valid_demo_ids:
                demo_id = demo_id.decode('utf-8')
                task_name = self.human_demo_dataset['data'][demo_id].attrs['task']
                
                if task_name not in self.task_demo_ids_maps:
                    self.task_demo_ids_maps[task_name] = []
                    
                self.task_demo_ids_maps[task_name].append(demo_id)
        else:
            path = self.json_path.replace('.json', "_indx_task_demo_map.json")
            with open(path, 'r') as f:
                self.task_demo_ids_maps = json.load(f)
        
        self.spawn_region = -1
        print("MimicPlay model loaded successfully.")

    def get_human_demo(self, env_name: str, variation_id: int, t: int, spawn_region: int):
        # Retrieve human demonstration for the given environment and variation
        # This is a placeholder implementation; replace with actual logic to get the demo
        goal_dict = {}
        
        demo_command = f"Human demo for {env_name} variation {variation_id}"
        
        task_name = f"task_{variation_id:02d}" 
        
        # remove the first element from the list
        if t == 0:
            if not self.same_conf and not self.same_spawn:
                # update current demo id at the beginning of the episode
                self.current_demo_id = self.task_demo_ids_maps[task_name].pop(0)
                self.task_demo_ids_maps[task_name].append(self.current_demo_id)
            else:
                self.current_demo_id = self.task_demo_ids_maps[str(spawn_region)][task_name]['train'].pop(0)
                self.task_demo_ids_maps[str(spawn_region)][task_name]['train'].append(self.current_demo_id)
            
        # get demo at index current_demo_id
        goal_obs_list = self.human_demo_dataset['data'][self.current_demo_id]['obs']['agentview_image']
        goal_image_length = goal_obs_list.shape[0]
        goal_indx = min(t+5, goal_image_length - 1) #+ 15
        
        goal_image = self.human_demo_dataset['data'][self.current_demo_id]['obs']['agentview_image'][goal_indx]
        
        goal_image = goal_image.astype('float32') / 255.0
        goal_image = goal_image.transpose(2, 0, 1)  # HWC to CHW
        # T x C x H x W
        goal_dict['agentview_image'] = goal_image  # add batch dimension
        
        return goal_dict

    def prepare_obs(self, obs, task_name='pick_place', gripper_closed=0):
        # Prepare observation for the policy
        # This is a placeholder implementation; replace with actual logic to prepare the obs
        
        policy_obs = {}
        for key in self.mimicplay_config['observation']['modalities']['obs']['rgb']:
            if 'agentview' in key:
                agent_view = crop_and_resize(
                    obs['camera_front_image'],
                    CROP_PARAMETERS[task_name],
                    TARGET_IMG_SIZE
                )
                agent_view = agent_view.astype('float32') / 255.0
                agent_view = agent_view.transpose(2, 0, 1)  # HWC to CHW
                # T x C x H x W
                policy_obs[key] = agent_view  # add time dimension
            elif 'eye_in_hand' in key:
                camera_gripper_image = cv2.flip(
                       obs['eye_in_hand_image'], 1
                    )
                camera_gripper_image = cv2.resize(
                    camera_gripper_image,
                    TARGET_IMG_SIZE,
                    interpolation=cv2.INTER_LINEAR
                )
                camera_gripper_image = camera_gripper_image.astype('float32') / 255.0
                camera_gripper_image = camera_gripper_image.transpose(2, 0, 1)  # HWC to CHW
                policy_obs[key] = camera_gripper_image  # add time dimension
            else:
                raise NotImplementedError(f"Unknown RGB observation key: {key}")
        
        low_level_keys = ["robot0_eef_pos_3d_world",
                          "robot0_eef_quat_world",
                          "robot0_gripper_qpos",
                          "robot0_eef_pos_3d_camera"]
        
        for key in low_level_keys:
            if 'eef_pos_3d_world' in key:
                policy_obs[key] = obs['eef_pos']
            elif 'eef_quat_world' in key:
                eef_quat_world = obs['eef_quat']
                R_ee_to_gripper = np.array([[0, -1, 0],
                                            [1,  0, 0],
                                            [0,  0, 1]])
                eef_mat = R_ee_to_gripper @ quat2mat(eef_quat_world)
                eef_euler = [normalize_angle(a) for a in mat2euler(eef_mat)]
                eef_quat_world = mat2quat(euler2mat(eef_euler))
                policy_obs[key] = eef_quat_world  # add time dimension
            elif 'gripper_qpos' in key:
                # policy_obs[key] = np.array([gripper_closed], dtype='float32')
                if gripper_closed:
                    policy_obs[key] = np.array([1.0], dtype='float32')  # closed
                elif not gripper_closed:
                    policy_obs[key] = np.array([-1.0], dtype='float32')  # open
            elif 'eef_pos_3d_camera' in key:
                #new_key = 'robot0_eef_pos_3D_0'
                new_key = 'robot0_eef_pos_3d_camera'
                policy_obs[new_key], _ = convert_from_world_to_camera_space(
                    pos=policy_obs['robot0_eef_pos_3d_world'],
                    quat=policy_obs['robot0_eef_quat_world'],
                    camera_pos=CAMERA_FRONT_POS,
                    camera_quat=CAMERA_FRONT_QUAT,
                    fov_y=fov_y,
                    img_width=obs['camera_front_image'].shape[1],
                    img_height=obs['camera_front_image'].shape[0],
                    img=obs['camera_front_image'],
                    t=0,
                    crop_params=CROP_PARAMETERS[task_name],
                    debug=False
                )
                policy_obs[new_key] = policy_obs[new_key]  # add time dimension
            else:
                raise NotImplementedError(f"Unknown low-dim observation key: {key}")
        return policy_obs

    def action_post_processing(self, obs, action_chunk, n_steps):
        # Post-process the action output by the policy
        post_processed_actions = []
        for action in action_chunk:
            # get current gripper position
            action = action #* SCALE_FACTOR  # Scale the action
            action_world = np.zeros(7)

            # round z position to 2 decimal places to avoid small fluctuations that can cause issues with grasping
            action[2] = round(action[2], 3)
            # print(f"Action after rounding: {action[2]}")
            action_world[0:3] = obs['eef_pos'] +(action[0:3])  # Scale delta position
            
            # Orientation action in world frame
            current_gripper_orientation =  mat2euler(R_EE_TO_GRIPPER @ quat2mat(obs['eef_quat']))
            current_gripper_orientation =  [normalize_angle(a) for a in current_gripper_orientation]
            gripper_orientation_action = current_gripper_orientation + action[3:6]
            gripper_orientation_action = [normalize_angle(a) for a in gripper_orientation_action]
            action_world[3:6] = euler_to_axis_angle(gripper_orientation_action)
            action_world[6] = action[-1]
            post_processed_actions.append(action_world)
        
        return post_processed_actions

    def compute_action(self, obs, resize_size, gripper_closed, task_description, task_name='pick_place', n_steps=-1, spawn_region=-1):
        # Compute action based
        
        # prepare observation
        policy_obs = self.prepare_obs(obs, 
                                      task_name, 
                                      gripper_closed)
        
        if n_steps == 0 and spawn_region == -1:
            # compute spawn region based on target object position
            target_obj_id = obs['target-object']
            target_obj_name = TARGET_BOX_ID_NAME_DICT["pick_place"][target_obj_id]
            target_obj_pos = obs[f"{target_obj_name}_pos"]
            self.spawn_region = get_spawn_region(target_obj_pos[1])
                    
        
        # get human obs
        human_obs = self.get_human_demo(env_name=task_name, 
                                        variation_id=task_description,
                                        t=n_steps,
                                        spawn_region=self.spawn_region)
        
        start_inf = time.time()
        action, guidance = self.policy(policy_obs, human_obs)
        end_inf = time.time()
        
        cur_img = np.array(policy_obs['agentview_image'].transpose(1, 2, 0) * 255.0).astype(np.uint8)
        act_out = guidance*120
        act_out = np.reshape(act_out, (10, 2))
        for i in range(act_out.shape[0]):
            pt = act_out[i]
            # print(pt)
            cur_img = cv2.circle(cur_img, (int(pt[1]), int(pt[0])), radius=5, color=(0, 255, 0), thickness=-1)
        
        
        human_demo = np.array(human_obs['agentview_image'].transpose(1, 2, 0) * 255.0).astype(np.uint8)
        # concatenate images side by side
        concat_img = np.concatenate((human_demo, cur_img), axis=1)
        pil_img = Image.fromarray(concat_img)
        pil_img.save(f"tmp/step_{n_steps}_concat.jpg")
        pil_img.save(f"step_concat.jpg")
        # print(f"Delta actions {action}")
        action = self.action_post_processing(
                                            obs=obs,
                                            action_chunk=[action], 
                                            n_steps=n_steps
                                          )
        
        return action, end_inf - start_inf, concat_img