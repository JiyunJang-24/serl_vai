import gym
from gym.spaces import flatten_space, flatten
import gym.spaces
import numpy as np

from scipy.spatial.transform import Rotation as R
from gym import Env
from gym import spaces
import time
import numpy as np
from pyquaternion import Quaternion


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": self.env.observation_space["state"],
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": obs["state"],
            **(obs["images"]),
        }
        return obs
    
    
class SERLObsSubGoalWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": self.env.observation_space["state"],
                **(self.env.observation_space["images"]),
                "subgoal": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.zeros(shape=(1,), dtype=np.int8).shape, dtype=np.int8
                ),
            }
        )

    def observation(self, obs):
        obs = {
            "state": obs["state"],
            **(obs["images"]),
            "subgoal": np.zeros(shape=(1,), dtype=np.int8)  # Assuming subgoal is a 1-dim vector
        }
        return obs
    
class SERLObsInstructWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": self.env.observation_space["state"],
                **(self.env.observation_space["images"]),
                "instruction": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=np.zeros(shape=(768,), dtype=np.float32).shape, dtype=np.float32
                ),
            }
        )

    def observation(self, obs):
        obs = {
            "state": obs["state"],
            **(obs["images"]),
            "instruction": np.zeros(shape=(768,), dtype=np.float32)  # Assuming instruction is a 768-dim vector
        }
        return obs


class SERLObsLIBEROWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, dense_reward=False):
        self.env = env
        obs_data = self.observation_()
        dummy_action = np.array([0.] * 7)
        self.dense_reward = dense_reward
        # 🔹 observation_space를 gym.spaces.Box 형태로 변환하여 gym.spaces.Dict에 전달
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["tcp_pose"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qpos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qpos"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qvel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qvel"].shape, dtype=np.float32
                        ),
                    }
                ),
                "agentview_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["agentview_image"].shape, dtype=np.uint8
                ),
                "robot0_eye_in_hand_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["robot0_eye_in_hand_image"].shape, dtype=np.uint8
                ),
            }
        )

        low = np.array([-0.75, -0.75, -0.75, -0.25, -0.25, -0.25, -1.0])
        high = np.array([ 0.75,  0.75,  0.75,  0.25,  0.25,  0.25,  1.0])
        self.action_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
        self.xyz_bounding_box = np.array([[-0.25, -0.1, 0.9], [0.15, 0.25, 1.2]])

        self.state_obs = None
    def observation_(self):
        obs = self.env.env._get_observations()
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos", "robot0_gripper_qvel"]
        other_states = {key: obs[key] for key in other_keys}
        
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        }

    def observation(self, obs):
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos", "robot0_gripper_qvel"]
        other_states = {key: obs[key] for key in other_keys}
        
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        }

    def step(self, action):

        # action = self.clip_safety_box(self.env.env._get_observations(), action)
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        info = None
        return self.observation(obs), info
        
    def render(self):
        return self.env.render()
    
    def clip_safety_box(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box while allowing movement back inside."""
        
        # 현재 엔드 이펙터 위치
        current_pos = obs['robot0_eef_pos']

        # 액션을 적용한 후 예상되는 위치
        next_pos = current_pos + (action[:3] / 100)

        # Safety Box 범위
        min_bounds, max_bounds = self.xyz_bounding_box  # (array([x_min, y_min, z_min]), array([x_max, y_max, z_max]))

        # Safety Box를 벗어나려는 방향 확인
        out_of_bounds_low = next_pos < min_bounds
        out_of_bounds_high = next_pos > max_bounds

        # **들어오는 액션인지 확인**
        moving_inward = (action[:3] > 0) & out_of_bounds_low | (action[:3] < 0) & out_of_bounds_high

        # Safety Box를 벗어나려 하고, 안으로 들어오는 방향이 아니라면 -> 이동 제한
        action[:3] = np.where(out_of_bounds_low & ~moving_inward, 0, action[:3])
        action[:3] = np.where(out_of_bounds_high & ~moving_inward, 0, action[:3])

        return action
    
    def get_state_obs(self):
        obs = self.env.env._get_observations()
        self.state_obs = {key: obs[key] for key in ['plate_1_pos', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_to_robot0_eef_pos']}
        return self.state_obs
    # def clip_safety_box(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    #     """Clip the pose to be within the safety box."""
    #     if (obs['robot0_eef_pos'] + (action[:3] / 100) < self.xyz_bounding_box[0]).sum() or (obs['robot0_eef_pos'][:3] + (action[:3] / 100) > self.xyz_bounding_box[1]).sum():
    #         action[:] = 0
    #     return action

class SERLObsVisualizationLIBEROWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        self.env = env
        obs_data = self.observation_()
        dummy_action = np.array([0.] * 7)

        # 🔹 observation_space를 gym.spaces.Box 형태로 변환하여 gym.spaces.Dict에 전달
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["tcp_pose"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qpos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qpos"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qvel": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qvel"].shape, dtype=np.float32
                        ),
                    }
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=dummy_action.shape, dtype=np.float32
        )

    def observation_(self):
        obs = self.env.env._get_observations()

        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos", "robot0_gripper_qvel"]
        other_states = {key: obs[key] for key in other_keys}
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            }
        }

    def observation(self, obs):
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos", "robot0_gripper_qvel"]
        other_states = {key: obs[key] for key in other_keys}
        
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            }
        }


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        info = None
        return self.observation(obs), info
        
    def render(self):
        return self.env.render()
    

class SERLObsRobosuiteWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, dense_reward=False):
        self.env = env
        obs_data = self.observation_()
        dummy_action = np.array([0.] * 7)
        self.dense_reward = dense_reward
        # 🔹 observation_space를 gym.spaces.Box 형태로 변환하여 gym.spaces.Dict에 전달
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["tcp_pose"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qpos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qpos"].shape, dtype=np.float32
                        ),
                    }
                ),
                "agentview_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["agentview_image"].shape, dtype=np.uint8
                ),
                "robot0_eye_in_hand_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["robot0_eye_in_hand_image"].shape, dtype=np.uint8
                ),

                "robot0_eye_in_hand_left_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["robot0_eye_in_hand_left_image"].shape, dtype=np.uint8
                ),
                #grasp_penalty np.float32, 1 dim
                # "grasp_penalty": gym.spaces.Box(
                #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                # ),
            }
        )
        self.dummy_action = np.array([0.] * 7)
        self.curr_gripper_pos = 0.04
        self.gripper_sleep = 0.1
        self.last_gripper_act = 0.0
        self.gripper_open_action = np.array([0, 0, 0, 0, 0, 0, -1])
        self.gripper_close_action = np.array([0, 0, 0, 0, 0, 0, 1])
        low = np.array([ -0.75,  -0.75,  -0.75,  0.0,  0.0,  0.0,  -1.0])
        high = np.array([ 0.75,  0.75,  0.75,  0.0,  0.0,  0.0,  1.0])
        self.action_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.xyz_bounding_box = np.array([[-0.12, -0.5, 0.8], [0.32, 0.55, 1.1]])
        fw_init_qpos = np.array([-0.11104011, 0.83457593, 0.11598836, -1.87966513, 0.03060678, 2.64441078, -0.84230056])

        # bw_init_qpos = np.array([0.33906145, 0.84236467, -0.01812242, -1.817323, -0.21512965, 2.63209568, 2.89666725])
        bw_init_qpos = np.array([0.03596594, 0.89493965, 0.2427182, -1.77591072, -0.1494936, 2.61021326, -0.46079463])

        

        self.env_task_to_init_qpos = [fw_init_qpos, bw_init_qpos]

        self.state_obs = None

    def observation_(self):
        obs = self.env._get_observations()
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos"]
        other_states = {key: obs[key] for key in other_keys}
        
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "robot0_eye_in_hand_left_image": obs["robot0_eye_in_hand_left_image"],
        }

    def observation(self, obs):
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos"]
        other_states = {key: obs[key] for key in other_keys}
        self.curr_gripper_pos = obs["robot0_gripper_qpos"][0]
        obs["robot0_eye_in_hand_image"] = np.rot90(obs["robot0_eye_in_hand_image"], 1)
        obs["robot0_eye_in_hand_left_image"] = np.rot90(obs["robot0_eye_in_hand_left_image"], 3)
        obs["agentview_image"] = np.rot90(obs["agentview_image"], 2)
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "robot0_eye_in_hand_left_image": obs["robot0_eye_in_hand_left_image"],
        }

    def step(self, action):

        
        if (action[-1] <= -0.5) and (self.curr_gripper_pos < 0.032):
            action[-1] = -1
            for i in range(7):
                obs, reward, done, info = self.env.step(self.gripper_open_action)

        elif (action[-1] >= 0.5) and (self.curr_gripper_pos > 0.032):
            action[-1] = 1
            for i in range(7):
                obs, reward, done, info = self.env.step(self.gripper_close_action)

        else:
            action[-1] = 0
        
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self, task_id = None):
        obs = self.env.reset()
        if task_id != None:
            self.env.robots[0].reset(deterministic=True, init_qpos_=self.env_task_to_init_qpos[task_id])
            obs, _, _, _ = self.env.step(self.dummy_action)
        info = None
        return self.observation(obs), info
        
    def render(self):
        return self.env.render()
    

    def get_state_obs(self):
        obs = self.env._get_observations()
        # self.state_obs = {key: obs[key] for key in ['plate_1_pos', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_to_robot0_eef_pos', 'Bread_pos']}
        self.state_obs = {key: obs[key] for key in ['Bread_pos', 'robot0_eef_pos']}
        return self.state_obs


class SERLObsRobosuiteInstructionWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, dense_reward=False):
        self.env = env
        obs_data = self.observation_()
        self.dense_reward = dense_reward
        # 🔹 observation_space를 gym.spaces.Box 형태로 변환하여 gym.spaces.Dict에 전달
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["tcp_pose"].shape, dtype=np.float32
                        ),
                        "robot0_gripper_qpos": gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=obs_data["state"]["robot0_gripper_qpos"].shape, dtype=np.float32
                        ),
                    }
                ),
                "agentview_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["agentview_image"].shape, dtype=np.uint8
                ),
                "robot0_eye_in_hand_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["robot0_eye_in_hand_image"].shape, dtype=np.uint8
                ),
                "robot0_eye_in_hand_left_image": gym.spaces.Box(
                    low=0, high=255, shape=obs_data["robot0_eye_in_hand_left_image"].shape, dtype=np.uint8
                ),
                #just 1 integer
                "instruction": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_data["instruction"].shape, dtype=np.float32
                ),
            }
        )
        self.dummy_action = np.array([0.] * 7)
        self.curr_gripper_pos = 0.04
        self.gripper_sleep = 0.1
        self.last_gripper_act = 0.0
        self.gripper_open_action = np.array([0, 0, 0, 0, 0, 0, -1])
        self.gripper_close_action = np.array([0, 0, 0, 0, 0, 0, 1])
        low = np.array([ -0.75,  -0.75,  -0.75,  0.0,  0.0,  0.0,  -1.0])
        high = np.array([ 0.75,  0.75,  0.75,  0.0,  0.0,  0.0,  1.0])

        self.action_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        
        self.xyz_bounding_box = np.array([[-0.12, -0.5, 0.8], [0.32, 0.55, 1.1]])

        fw_init_qpos = np.array([-0.11104011, 0.83457593, 0.11598836, -1.87966513, 0.03060678, 2.64441078, -0.84230056])
        # bw_init_qpos = np.array([0.33906145, 0.84236467, -0.01812242, -1.817323, -0.21512965, 2.63209568, 2.89666725])
        bw_init_qpos = np.array([0.03596594, 0.89493965, 0.2427182, -1.77591072, -0.1494936, 2.61021326, -0.46079463])

        
        self.env_task_to_init_qpos = [fw_init_qpos, bw_init_qpos]

        self.state_obs = None
    def observation_(self):
        obs = self.env._get_observations()
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos"]
        other_states = {key: obs[key] for key in other_keys}
        
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "robot0_eye_in_hand_left_image": obs["robot0_eye_in_hand_left_image"],
            "instruction": np.zeros(shape=(768,), dtype=np.float32)

        }

    def observation(self, obs):
        # 🔹 TCP Pose (xyz + quat) 그룹화
        tcp_pose_keys = ["robot0_eef_pos", "robot0_eef_quat"]
        tcp_pose = np.concatenate([obs[key] for key in tcp_pose_keys], axis=-1)
        # 🔹 기타 상태 정보 유지
        other_keys = ["robot0_gripper_qpos"]
        other_states = {key: obs[key] for key in other_keys}
        self.curr_gripper_pos = obs["robot0_gripper_qpos"][0]
        obs["robot0_eye_in_hand_image"] = np.rot90(obs["robot0_eye_in_hand_image"], 1)
        obs["robot0_eye_in_hand_left_image"] = np.rot90(obs["robot0_eye_in_hand_left_image"], 3)
        obs["agentview_image"] = np.rot90(obs["agentview_image"], 2)
        return {
            "state": {
                "tcp_pose": tcp_pose,
                **other_states,  # 기존 키 유지
            },
            "agentview_image": obs["agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "robot0_eye_in_hand_left_image": obs["robot0_eye_in_hand_left_image"],
            "instruction": np.zeros(shape=(768,), dtype=np.float32)
        }
    def step(self, action):

        # action = self.clip_safety_box(self.env._get_observations(), action)
        #clip using action low, high
        #multiply exist action and high
        # action = action * self.action_space.high
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if (action[-1] <= -0.5) and (self.curr_gripper_pos < 0.032):
            action[-1] = -1
            for i in range(7):
                obs, reward, done, info = self.env.step(self.gripper_open_action)

        elif (action[-1] >= 0.5) and (self.curr_gripper_pos > 0.032):
            action[-1] = 1
            for i in range(7):
                obs, reward, done, info = self.env.step(self.gripper_close_action)

        else:
            action[-1] = 0

        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self, task_id = None):
        obs = self.env.reset()
        if task_id != None:
            self.env.robots[0].reset(deterministic=True, init_qpos_=self.env_task_to_init_qpos[task_id])
            obs, _, _, _ = self.env.step(self.dummy_action)
        info = None
        return self.observation(obs), info
        
    def render(self):
        return self.env.render()
    
    def clip_safety_box(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box while allowing movement back inside."""
        
        # 현재 엔드 이펙터 위치
        current_pos = obs['robot0_eef_pos']

        # 액션을 적용한 후 예상되는 위치
        next_pos = current_pos + (action[:3] / 20)

        # Safety Box 범위
        min_bounds, max_bounds = self.xyz_bounding_box  # (array([x_min, y_min, z_min]), array([x_max, y_max, z_max]))

        # Safety Box를 벗어나려는 방향 확인
        out_of_bounds_low = next_pos < min_bounds
        out_of_bounds_high = next_pos > max_bounds

        # **들어오는 액션인지 확인**
        moving_inward = ((action[:3] > 0) & out_of_bounds_low) | ((action[:3] < 0) & out_of_bounds_high)

        # Safety Box를 벗어나려 하고, 안으로 들어오는 방향이 아니라면 -> 이동 제한
        action[:3] = np.where(out_of_bounds_low & ~moving_inward, 0, action[:3])
        action[:3] = np.where(out_of_bounds_high & ~moving_inward, 0, action[:3])

        return action
    
    def get_state_obs(self):
        obs = self.env._get_observations()
        # self.state_obs = {key: obs[key] for key in ['plate_1_pos', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_to_robot0_eef_pos', 'Bread_pos']}
        self.state_obs = {key: obs[key] for key in ['Bread_pos', 'robot0_eef_pos']}
        return self.state_obs




class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...).
    By convention, the 7th dimension of the action space is used for the gripper.

    """

    def __init__(self, env: Env, include_relative_pose=True):
        self.env = env
        if self.action_space.shape == (4,):
            self.only_pos_control = True
            self.fixed_rotation = None
        else:
            self.only_pos_control = False
        
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        if self.only_pos_control:
        # [x, y, z, gripper] → [x, y, z, 0, 0, 0, gripper]
            action = np.concatenate([action[:3], np.zeros(3), action[3:]])
            
        transformed_action = self.transform_action(action)
        if self.only_pos_control:
            transformed_action = np.concatenate([transformed_action[:3], np.array([transformed_action[-1]])])

        obs, reward, done, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            if self.only_pos_control:
                info["intervene_action"] = np.concatenate([info["intervene_action"][:3], np.zeros(3), info["intervene_action"][3:]])
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )
            if self.only_pos_control:
                info["intervene_action"] = np.concatenate([info["intervene_action"][:3], np.array([info["intervene_action"][-1]])])
        if self.only_pos_control:
            obs["state"]["tcp_pose"] = np.concatenate((obs["state"]["tcp_pose"], self.fixed_rotation))
            obs["state"]["tcp_vel"] = np.concatenate((obs["state"]["tcp_vel"], np.zeros(3)))
        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Update adjoint matrix
        reset_pose = info['reset_pose']
        
        if self.only_pos_control:
            self.fixed_rotation = reset_pose[3:]
            obs["state"]["tcp_pose"] = np.concatenate((obs["state"]["tcp_pose"], reset_pose[3:]))
            obs["state"]["tcp_vel"] = np.concatenate((obs["state"]["tcp_vel"], np.zeros(3)))

        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        if self.only_pos_control:
            obs["state"]["tcp_pose"] = obs["state"]["tcp_pose"][:3]
            obs["state"]["tcp_vel"] = obs["state"]["tcp_vel"][:3]

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action
    

class RelativeRobosuiteFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...).
    By convention, the 7th dimension of the action space is used for the gripper.

    """

    def __init__(self, env: Env, include_relative_pose=True):
        self.env = env
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        
            
        transformed_action = self.transform_action(action)
        
        obs, reward, done, info = self.env.step(transformed_action)
        print(obs['state']['tcp_pose'])
        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )
        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Update adjoint matrix        
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        # obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T



class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        self.env = env
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements



class ScaleObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    (to somewhat normalize the observations space)
    """

    def __init__(self,
                 env,
                 translation_scale=100.,
                 rotation_scale=10.,
                 force_scale=1.,
                 torque_scale=10.
                 ):
        super().__init__(env)
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.force_scale = force_scale
        self.torque_scale = torque_scale

    def scale_wrapper_get_scales(self):
        return dict(
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            force_scale=self.force_scale,
            torque_scale=self.torque_scale
        )

    def observation(self, obs):
        if obs["state"]["tcp_pose"].shape == (3,):
            obs["state"]["tcp_pose"][:3] *= self.translation_scale
            obs["state"]["tcp_vel"][:3] *= self.translation_scale
        else:
            obs["state"]["tcp_pose"][:3] *= self.translation_scale
            obs["state"]["tcp_vel"][:3] *= self.translation_scale
            obs["state"]["tcp_pose"][3:] *= self.rotation_scale
            obs["state"]["tcp_vel"][3:] *= self.rotation_scale
        obs["state"]["tcp_force"] *= self.force_scale
        obs["state"]["tcp_torque"] *= self.torque_scale
        return obs

class SafeControlWrapper(gym.ObservationWrapper):
    """
    This wrapper ensures that the robot does not enter the inner safety box.
    If the target pose is inside the inner bounding box, it clips to the intersection point.
    """

    def __init__(self, env):
        self.env = env
        self.current = None
        self.intersection = None
        INNER_ABS_POSE_LIMIT_LOW = np.array([0.045, -0.017, 0.4, 2.86, -0.1, -3.14])
        INNER_ABS_POSE_LIMIT_HIGH = np.array([0.15, 0.027, 0.86, 4, 0.1, 3.14])
        self.inner_bounding_box = gym.spaces.Box(
            INNER_ABS_POSE_LIMIT_LOW[:3],
            INNER_ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.obs = None
        self.action_scale = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01])  # scale for action space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current = obs["state"]["tcp_pose"][:3]
        self.intersection = None
        self.obs = obs
        return obs, info
    
    def step(self, action):
        """
        Clip the action to be within the safety box while allowing movement back inside.
        """
        curr_pose = self.obs["state"]["tcp_pose"]
        
        # action = self.clip_safety_box(self.env._get_observations(), action)
        # clip using action low, high
        # multiply exist action and high
        # action = action * self.action_space.high
        # curr_obs = self.observation_()['state']['tcp_pose']
        action = action * self.action_scale
        safe_pos_euler = curr_pose.copy()
        safe_pos_euler[:3] = curr_pose[:3] + action[:3]
        safe_pos_euler = self.apply_inner_box_projection(safe_pos_euler, curr_pose)
        action[:3] = safe_pos_euler[:3] - curr_pose[:3]
        
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, done, info = self.env.step(action)
        self.obs = obs
        return obs, reward, done, info
    
    def clip_safety_box(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box while allowing movement back inside."""
        
        # 현재 엔드 이펙터 위치
        current_pos = obs['robot0_eef_pos']

        # 액션을 적용한 후 예상되는 위치
        next_pos = current_pos + (action[:3] / 20)

        # Safety Box 범위
        min_bounds, max_bounds = self.xyz_bounding_box  # (array([x_min, y_min, z_min]), array([x_max, y_max, z_max]))

        # Safety Box를 벗어나려는 방향 확인
        out_of_bounds_low = next_pos < min_bounds
        out_of_bounds_high = next_pos > max_bounds

        # **들어오는 액션인지 확인**
        moving_inward = ((action[:3] > 0) & out_of_bounds_low) | ((action[:3] < 0) & out_of_bounds_high)

        # Safety Box를 벗어나려 하고, 안으로 들어오는 방향이 아니라면 -> 이동 제한
        action[:3] = np.where(out_of_bounds_low & ~moving_inward, 0, action[:3])
        action[:3] = np.where(out_of_bounds_high & ~moving_inward, 0, action[:3])

        return action

    def intersect_line_bbox(self, p1, p2, bbox_min, bbox_max):
        # Define the parameterized line segment
        # P(t) = p1 + t(p2 - p1)
        tmin = 0
        tmax = 1

        for i in range(3):
            if p1[i] < bbox_min[i] and p2[i] < bbox_min[i]:
                return None
            if p1[i] > bbox_max[i] and p2[i] > bbox_max[i]:
                return None

            # For each axis (x, y, z), compute t values at the intersection points
            if abs(p2[i] - p1[i]) > 1e-10:  # To prevent division by zero
                t1 = (bbox_min[i] - p1[i]) / (p2[i] - p1[i])
                t2 = (bbox_max[i] - p1[i]) / (p2[i] - p1[i])

                # Ensure t1 is smaller than t2
                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Compute the intersection point using the t value
        intersection = p1 + tmin * (p2 - p1)

        return intersection
    
    def intersect_line_bbox_proj(self, p1, p2, bbox_min, bbox_max):
        """
        Return the first intersection point of line segment [p1, p2] with the bounding box.
        If the line does not intersect, return None.
        """
        tmin = 0.0
        tmax = 1.0

        direction = p2 - p1

        for i in range(3):
            if abs(direction[i]) < 1e-10:
                # Line is parallel to slab. No hit if origin not within slab
                if p1[i] < bbox_min[i] or p1[i] > bbox_max[i]:
                    return None
            else:
                t1 = (bbox_min[i] - p1[i]) / direction[i]
                t2 = (bbox_max[i] - p1[i]) / direction[i]

                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        # Confirm intersection lies within the segment
        if tmin > 1.0 or tmax < 0.0:
            return None

        return p1 + tmin * direction


    def apply_inner_box_clipping(self, pose: np.ndarray, curr_pose:np.ndarray) -> np.ndarray:
        """
        Prevent the gripper from entering the inner safety box.
        If the target pose is inside the box, clip to the intersection point.
        """
        if self.inner_bounding_box.contains(pose[:3]):
            clipped_xyz = self.intersect_line_bbox(
                curr_pose[:3],
                pose[:3],
                self.inner_bounding_box.low,
                self.inner_bounding_box.high,
            )
            if clipped_xyz is not None:
                pose[:3] = clipped_xyz
        return pose

    def clamp_to_inner_box_surface(self, current, target, epsilon=1e-4):
        direction = target - current
        t_candidates = []

        for i in range(3):  # x, y, z
            if abs(direction[i]) < 1e-8:
                continue  # No movement along this axis

            if direction[i] > 0 and current[i] < self.inner_bounding_box.low[i] and target[i] > self.inner_bounding_box.low[i]:
                t = (self.inner_bounding_box.low[i] - current[i]) / direction[i]
                t_candidates.append(t)
            elif direction[i] < 0 and current[i] > self.inner_bounding_box.high[i] and target[i] < self.inner_bounding_box.high[i]:
                t = (self.inner_bounding_box.high[i] - current[i]) / direction[i]
                t_candidates.append(t)

        if not t_candidates:
            return target  # No intersection → safe

        t_min = min(t_candidates)
        t_min = max(0.0, min(t_min, 1.0))  # clamp t to [0,1]
        new_target = current + t_min * direction

        return new_target

    def apply_inner_box_projection(self, pose: np.ndarray, curr_pose: np.ndarray) -> np.ndarray:
        """
        If target pose is inside the inner bounding box, project it to the face of the box
        intersected by the line from current pose, so the final pose stays on the surface.
        """
        target = pose[:3].copy()
        current = curr_pose[:3]
        if self.inner_bounding_box.contains(current):
            if self.current is None:
                pose[:3] = target
            else:
                pose[:3] = self.current[:3]
                # pose[:3] = self.intersection[:3]
        else:
            self.current = current
            
            # if self.inner_bounding_box.contains(target):
            intersection = self.intersect_line_bbox(
                current, target,
                self.inner_bounding_box.low,
                self.inner_bounding_box.high
            )
            
            if intersection is not None:
                self.intersection = intersection
                epsilon = 1e-4
                # Determine which face was intersected
                for i in range(3):  # x, y, z
                    if abs(intersection[i] - self.inner_bounding_box.low[i]) < epsilon:
                        target[i] = self.inner_bounding_box_move.low[i] - 1e-3
                        target[i] = intersection[i]
                        continue
                    elif abs(intersection[i] - self.inner_bounding_box.high[i]) < epsilon:
                        target[i] = self.inner_bounding_box_move.high[i] + 1e-3
                        target[i] = intersection[i]
                        continue
                self.current = target
                pose[:3] = target  # apply modified target
        return pose
