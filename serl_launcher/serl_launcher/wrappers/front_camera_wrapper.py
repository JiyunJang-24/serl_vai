import gym
from gym.core import Env, spaces
from copy import deepcopy

import time
import numpy as np
from gym.spaces import Box
import copy
sigmoid = lambda x: 1 / (1 + np.exp(-x))
import jax
import cv2
from collections import deque
import flax.linen as nn

class FrontCameraWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        front_obs_space = {
            k: space for k, space in self.observation_space.items() if "front" in k
        }
        wrist_obs_space = {
            k: space for k, space in self.observation_space.items() if "wrist" in k
        }
        self.front_observation_space = gym.spaces.Dict(front_obs_space)
        self.wrist_observation_space = gym.spaces.Dict(wrist_obs_space)
        # self.observation_space = gym.spaces.Dict(new_obs_space)
        self.front_obs = None

    def observation(self, observation):
        # cache a copy of observation with only the front camera image
        new_obs = deepcopy(observation)
        if "wrist" in new_obs:
            new_obs.pop("wrist")
        if "state" in new_obs:
            new_obs.pop("state")
        if "instruction" in new_obs:
            new_obs.pop("instruction")
        if "subgoal" in new_obs:
            new_obs.pop("subgoal")
        self.front_obs = new_obs

        return observation

    def get_front_cam_obs(self):
        return self.front_obs


class FrontCameraLIBEROWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env):
        # super().__init__(env)
        self.env = env
        front_obs_space = {
            k: space for k, space in self.env.observation_space.items() if "agentview_image" == k
        }
        self.front_image_keys = ["agentview_image"]
        self.front_observation_space = gym.spaces.Dict(front_obs_space)
        # self.observation_space = gym.spaces.Dict(new_obs_space)

        image_obs_space = {
            k: space for k, space in self.env.observation_space.items() if "state" != k and "instruction" != k
        }
        self.image_keys = ["agentview_image", "robot0_eye_in_hand_image", "robot0_eye_in_hand_left_image"]
        self.image_observation_space = gym.spaces.Dict(image_obs_space)
        # self.observation_space = gym.spaces.Dict(new_obs_space)
        hand_image_obs_space = {
            k: space for k, space in self.env.observation_space.items() if "hand" in k
        }
        self.hand_image_observation_space = gym.spaces.Dict(hand_image_obs_space)
        self.hand_image_keys = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_left_image"]
        self.image_obs = None

        self.state_obs = None

    def observation(self, observation):
        # cache a copy of observation with only the front camera image
        # observation['agentview_image'] = np.rot90(observation['agentview_image'], k=2)

        new_obs = deepcopy(observation)
        new_obs.pop("state")
        if "instruction" in new_obs:
            new_obs.pop("instruction")
        image_obs = deepcopy(new_obs)
        self.image_obs = image_obs
        if "robot0_eye_in_hand_image" in new_obs:
            new_obs.pop("robot0_eye_in_hand_image")
        if "robot0_eye_in_hand_left_image" in new_obs:
            new_obs.pop("robot0_eye_in_hand_left_image")
        self.front_obs = new_obs

        new_obs2 = deepcopy(observation)
        if "robot0_eye_in_hand_left_image" in new_obs2:
            new_obs2.pop("robot0_eye_in_hand_left_image")
        if "robot0_eye_in_hand_image" in new_obs2:
            new_obs2.pop("robot0_eye_in_hand_image")
        new_obs2.pop("agentview_image")
        if "instruction" in new_obs2:
            new_obs2.pop("instruction")
    
        self.state_obs = new_obs2

        return observation

    def get_front_cam_obs(self):
        return self.front_obs
    
    def get_image_obs(self):
        return self.image_obs
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    
    def get_hand_image_obs(self):
        image = self.get_image_obs()
        image.pop("agentview_image")
        return image

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info


class FWBWFrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, fw_reward_classifier_func, bw_reward_classifier_func):

        self.env = env
        self.task_id = 0
        
        self.reward_classifier_funcs = [
            fw_reward_classifier_func,
            bw_reward_classifier_func,
        ]

    def set_task_id(self, task_id):
        self.task_id = task_id

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_reward(self, obs, info):
        
        reward = self.reward_classifier_funcs[self.task_id](obs).item()
        gripper_open = obs['state']['gripper_state'] > 0.6
        # if self.task_id:
        #     success_pos = obs['state']['tcp_pose'][1] > 0.15
        # else:
        #     success_pos = obs['state']['tcp_pose'][1] < -0.13
        # curr_pos = obs['state'][5:11]
        # curr_pos_y = obs['state'][6]
        
        info['real_reward'] = sigmoid(reward)
        return ((sigmoid(reward) >= 0.9) * 1) and gripper_open

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        
        success = self.compute_reward(obs, info)
        rew = int(success)
        done = success
        return obs, rew, done, info
    
    def reset(self, **kwargs):
        if "task_id" in kwargs:
            self.set_task_id(kwargs["task_id"])
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def relabel_obs(self, obs, info):
        success = self.compute_reward(obs, info)
        reward = int(success)
        done = success
        return reward, done
    
def is_grasping_bread(robot_pos, gripper_qpos, bread_pos, tcp_thresh=0.032, xy_thresh=0.035, height_thresh=0.32, gripper_close_thresh=0.06):
    """
    로봇이 빵을 잡았는지 여부를 반환.
    
    Args:
        tcp_pose: (7,) array. End-effector의 pose (x, y, z, qx, qy, qz, qw)
        gripper_qpos: (2,) array. 그리퍼의 관절 위치
        bread_pos: (3,) array. 빵의 중심 좌표
        tcp_thresh: 빵 중심과 tcp 사이의 허용 거리
        height_thresh: 그리퍼가 빵 위에 어느 정도 떠 있어야 함
        gripper_close_thresh: 그리퍼가 어느 정도 닫혀 있어야 함
    """
    tcp_pose = robot_pos
    gripper_qpos = gripper_qpos
    tcp_pos = tcp_pose[:3]
    tcp_bread_dist = np.linalg.norm(tcp_pos - bread_pos)

                # xy 평면 거리 차이 (개별 축 기준)
    xy_dist = np.linalg.norm(tcp_pos[:2] - bread_pos[:2])
    is_xy_near = xy_dist < xy_thresh
    
    is_near = tcp_bread_dist < tcp_thresh and is_xy_near
    is_above = bread_pos[2] > 0.85 and tcp_pose[2] > 0.85 and (tcp_pos[2] - bread_pos[2]) < height_thresh
    
    is_gripper_closed = (np.abs(gripper_qpos[0] - gripper_qpos[1]) <= gripper_close_thresh) and (np.abs(gripper_qpos[0] - gripper_qpos[1]) >= 0.0373)
    # print("is near: ", xy_dist, is_near, "is above:", (tcp_pos[2] - bread_pos[2]), is_above, "is gripper:", (np.abs(gripper_qpos[0] - gripper_qpos[1]), is_gripper_closed))
    return is_near and is_gripper_closed


class FWBWFrontCameraRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, fw_reward_classifier_func, bw_reward_classifier_func):

        self.env = env
        self.task_id = 0
        
        self.reward_classifier_funcs = [
            fw_reward_classifier_func,
            bw_reward_classifier_func,
        ]
        self.grasp_success = 0
    def set_task_id(self, task_id):
        self.task_id = task_id

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        reward, success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_reward(self, obs):
        state_obs = self.get_state_obs()
        bread_pos = state_obs['Bread_pos']
        obs_robot_pos = state_obs['robot0_eef_pos']

        if is_grasping_bread(obs_robot_pos, obs['state']['robot0_gripper_qpos'], bread_pos):
            self.grasp_success = 1
            grasp_reward = 0.5
            if self.task_id:
                penalty = -np.abs(obs_robot_pos[1] - (-0.07))
                penalty_bread = -np.abs(bread_pos[1] - (-0.07))
            else:
                penalty = -np.abs(obs_robot_pos[1] - 0.07)
                penalty_bread = -np.abs(bread_pos[1] - 0.07)

            penalty += -np.abs(obs_robot_pos[0] - 0.095)
            penalty += -np.abs(np.clip(obs_robot_pos[2], 0, 0.88) - 0.88)
            grasp_reward += penalty
            grasp_reward += penalty_bread
        else:
            self.grasp_success = 0
            grasp_reward = -np.linalg.norm(bread_pos - obs_robot_pos[:3]) * 5.0

        if self.task_id:
            success_box = np.array([[0.005, -0.14, 0.81], [0.19, -0.03, 0.9]])
            success_check = (success_box[0, 0] < state_obs['Bread_pos'][0] < success_box[1, 0]) and (success_box[0, 1] < state_obs['Bread_pos'][1] < success_box[1, 1]) and (success_box[0, 2] < state_obs['Bread_pos'][2] < success_box[1, 2])
        else:
            success_box = np.array([[0.005, 0.05, 0.81], [0.19, 0.14, 0.9]])
            success_check = (success_box[0, 0] < state_obs['Bread_pos'][0] < success_box[1, 0]) and (success_box[0, 1] < state_obs['Bread_pos'][1] < success_box[1, 1]) and (success_box[0, 2] < state_obs['Bread_pos'][2] < success_box[1, 2])


        if success_check:
            reward = 1.0
            success = 1.0
        else:
            reward = 0.0
            success = 0.0
            reward = reward + grasp_reward
        # reward = self.reward_classifier_funcs[self.task_id](obs).item()
        # reward = nn.sigmoid(reward)
        
        # if (reward >= 0.9):
        #     state_obs = self.get_state_obs()
        #     if self.task_id:
        #         success_box = np.array([[0.005, -0.14, 0.81], [0.19, -0.025, 0.89]])
        #         success_check = (success_box[0, 0] < state_obs['Bread_pos'][0] < success_box[1, 0]) and (success_box[0, 1] < state_obs['Bread_pos'][1] < success_box[1, 1]) and (success_box[0, 2] < state_obs['Bread_pos'][2] < success_box[1, 2])
        #     else:
        #         success_box = np.array([[0.005, 0.03, 0.81], [0.19, 0.14, 0.89]])
        #         success_check = (success_box[0, 0] < state_obs['Bread_pos'][0] < success_box[1, 0]) and (success_box[0, 1] < state_obs['Bread_pos'][1] < success_box[1, 1]) and (success_box[0, 2] < state_obs['Bread_pos'][2] < success_box[1, 2])
        #     success = (reward >= 0.9) * 1
        #     success = success and success_check
        #     if success:
        #         reward = 1
        #     else:
        #         reward = 0
        # else:
        #     reward = 0
        #     success = 0

        
        return reward, success

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        reward, success = self.compute_reward(obs)
        info['grasp_success'] = self.grasp_success
        return obs, reward, success, info
    

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def get_state_obs(self):
        return self.env.get_state_obs()



class FrontCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        success = self.compute_reward(self.env.get_front_cam_obs())
        rew += success
        done = done or success
        return obs, rew, done, truncated, info


class BinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    """

    def __init__(self, env: Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        success = self.compute_reward(obs)
        rew += success
        done = done or success
        return obs, rew, done, truncated, info






class GraspClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, grasp_reward_classifier_func):

        self.env = env
        self.grasp = 0
        self.task_id = 0
        self.grasp_queue = deque([False] * 5, maxlen=5)  # 최근 grasp 상태 저장
        self.grasp_reward_classifier_func = grasp_reward_classifier_func

    def set_task_id(self, task_id):
        self.task_id = task_id
        self.env.set_task_id(task_id)
        

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        reward, success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_grasp(self, obs, info):
        rew = self.grasp_reward_classifier_func(obs).item()
        grasp_prob = nn.sigmoid(rew) > 0.9
        info['grasp_prob'] = [grasp_prob, False, all(self.grasp_queue)]
        grip = obs['state']['gripper_state'] > 0.1 and obs['state']['gripper_state'] < 0.45
        # print(obs['state']['gripper_state'])
        info['grasp_prob'][1] = grip
        real_grasp = info['grasp_prob'][0] and grip
        if real_grasp:
            past_grasp = all(self.grasp_queue)
            if past_grasp:
                reward = 0.3
            else:
                reward = 0.0
        else:
            reward = 0.0
        self.grasp_queue.append(real_grasp)
        return reward

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        reward = self.compute_grasp(obs, info)
        if done:
            reward = 1.0
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    
    def relabel_obs(self, obs, info):
        reward, done = self.env.relabel_obs(obs, {})
        reward = self.compute_grasp(obs, {})
        if done:
            reward = 1.0
        return reward, done


class GraspClassifierNoVisionWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env):
        self.env = env
        self.grasp = 0
        self.task_id = 0
        self.grasp_queue = deque([False] * 5, maxlen=5)  # 최근 grasp 상태 저장

    def set_task_id(self, task_id):
        self.task_id = task_id
        # self.env.set_task_id(task_id)

    
    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        reward, success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_grasp(self, obs, info):
        info['grasp_prob'] = [False, all(self.grasp_queue)]
        grip = obs['state']['gripper_state'] > 0.1 and obs['state']['gripper_state'] < 0.45
        # print(obs['state']['gripper_state'])
        info['grasp_prob'][0] = grip
        real_grasp = grip
        if real_grasp:
            past_grasp = all(self.grasp_queue)
            if past_grasp:
                reward = 0.15
            else:
                reward = 0.0
        else:
            reward = 0.0
        self.grasp_queue.append(real_grasp)
        return reward

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        reward = self.compute_grasp(obs, info)
        # done = False
        if done:
            reward = 1.0
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.grasp_queue = deque([False] * 5, maxlen=5)  # 최근 grasp 상태 저장
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    
    def relabel_obs(self, obs, info):
        reward, done = self.env.relabel_obs(obs, {})
        reward = self.compute_grasp(obs, {})
        if done:
            reward = 1.0
        return reward, done
        



class GraspClassifierRobosuiteWrapper(gym.Wrapper):
    """
    This wrapper uses the front camera images to compute the reward,
    which is not part of the RL policy's observation space. This is used for the
    forward backward reset-free bin picking task, where there are two classifiers,
    one for classifying success + failure for the forward and one for the
    backward task. Here we also use these two classifiers to decide which
    task to transition into next at the end of the episode to maximize the
    learning efficiency.
    """

    def __init__(self, env: Env, grasp_reward_classifier_func):

        self.env = env
        self.grasp = 0
        self.task_id = 0
        self.grasp_reward_classifier_func = grasp_reward_classifier_func

    def set_task_id(self, task_id):
        self.task_id = task_id
        self.env.set_task_id(task_id)
        

    def task_graph(self, obs):
        """
        predict the next task to transition into based on the current observation
        if the current task is not successful, stay in the current task
        else transition to the next task
        """
        reward, success = self.compute_reward(obs)
        if success:
            return (self.task_id + 1) % 2
        return self.task_id

    def compute_grasp(self, image_obs, obs):
        rew = self.grasp_reward_classifier_func(image_obs).item()
        
        grasp_prob = nn.sigmoid(rew)
        bread_pos = self.get_state_obs()['Bread_pos']
        if (grasp_prob >= 0.95):
            self.grasp = 1
        else:
            self.grasp = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.compute_grasp(self.env.env.get_hand_image_obs(), obs)
        # state_obs = self.get_state_obs()
        grasp_bonus = 0.3 if self.grasp else 0.0  # Grasp 성공 시 보너스
        # distance = np.linalg.norm(state_obs['Bread_to_robot0_eef_pos'])

        # 거리 감소 시 리워드 증가하도록 음수 처리
        reward = rew + grasp_bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None  # 실제로는 robot0_gripper_qpos[0]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"]["robot0_gripper_qpos"][0]
        return obs, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        current_gripper_pos = self.last_gripper_pos
        act = action[-1]

        is_already_open = current_gripper_pos > 0.036  # 완전히 열린 상태 기준 (0.04 근처)
        is_already_closed = current_gripper_pos < 0.036  # 완전히 닫힌 상태 기준 (0.0 근처)

        if (act < -0.5 and is_already_closed) or (act > 0.5 and is_already_open):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"]["robot0_gripper_qpos"][0]
        return observation, reward, done, info
    
class GripperPenaltyUR5Wrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None 

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"]["gripper_state"]
        return obs, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        current_gripper_pos = self.last_gripper_pos
        act = action[-1]

        is_already_open = current_gripper_pos > 0.6  # 완전히 열린 상태 기준 (0.04 근처)
        is_already_closed = current_gripper_pos < 0.6  # 완전히 닫힌 상태 기준 (0.0 근처)
        # act > 0.5면 그리퍼를 닫는 액션
        # 의미있는 그리퍼 액션에 대한 페널티
        if (act < -0.5 and is_already_closed) or (act > 0.5 and is_already_open):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"]["gripper_state"]
        return observation, reward, done, info