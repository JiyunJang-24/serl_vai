import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import flax.linen as nn
from scipy.spatial.transform import Rotation as R

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper, SERLObsRobosuiteWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierNoVisionWrapper, GraspClassifierWrapper, GraspClassifierRobosuiteWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO, SpacemouseInterventionUR5
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark
from libero.libero.envs import *
from libero.libero.envs import OffScreenRenderEnv
import argparse
import ur_env
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import matplotlib.pyplot as plt
import jax
import cv2
from collections import deque



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", type=str, required=True, help="Name of the demo file (e.g., bw_pick~.pkl)")
    parser.add_argument("--backward", type=int, default=0, help="backward")
    parser.add_argument("--fw_reward_classifier_ckpt_path", type=str, default=None, required=True, help="Path to the forward reward classifier checkpoint")
    parser.add_argument("--bw_reward_classifier_ckpt_path", type=str, default=None, required=True, help="Path to the backward reward classifier checkpoint")
    parser.add_argument("--grasp_reward_classifier_ckpt_path", type=str, default=None, required=False, help="Path to the grasp reward classifier checkpoint")
    args = parser.parse_args()

    demo_load_path = os.path.join("demos", args.demo_file)
    # demo_initial_path = os.path.join("demos", "initial_state/"+args.demo_file)
    demo_save_path = os.path.join("relabel_demos", args.demo_file)

    # Load demo data
    if not os.path.exists(demo_load_path):
        raise FileNotFoundError(f"Demo file not found: {demo_load_path}")
    
    # if not os.path.exists(demo_initial_path):
    #     raise FileNotFoundError(f"Demo initial file not found: {demo_initial_path}")

    with open(demo_load_path, "rb") as f:
        demo_data = pkl.load(f)

    env = gym.make("box_picking_camera_env",
                   camera_mode="rgb",
                   max_episode_length=200,
                   save_video=True,
                   fake_env=True
                   )
    env = SpacemouseInterventionUR5(env, fake_env=True)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    # env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    
    front_image_keys = [
        k for k in env.front_observation_space.keys() if "state" not in k
    ]
    wrist_image_keys = [
        k for k in env.wrist_observation_space.keys() if "state" not in k
    ]
    from serl_launcher.networks.reward_classifier import load_classifier_func
    fw_reward_classfier_ckpt_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/fw_checkpoint_50000"
    bw_reward_classfier_ckpt_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/bw_checkpoint_50000"
    grasp_classifier_ckpt_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/grasp_checkpoint_30000"
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    if (
        not fw_reward_classfier_ckpt_path
        or not bw_reward_classfier_ckpt_path
    ):
        raise ValueError(
            "Must provide both fw and bw reward classifier ckpt paths for actor"
        )
    fw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=front_image_keys,
        checkpoint_path=fw_reward_classfier_ckpt_path,
    )
    rng, key = jax.random.split(rng)
    bw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=front_image_keys,
        checkpoint_path=bw_reward_classfier_ckpt_path,
    )
    env = FWBWFrontCameraBinaryRewardClassifierWrapper(
        env, fw_classifier_func, bw_classifier_func
    )
    # grasp_classifier_func = load_classifier_func(
    #     key=key,
    #     sample=env.wrist_observation_space.sample(),
    #     image_keys=wrist_image_keys,
    #     checkpoint_path=grasp_classifier_ckpt_path,
    # )
    # env = GraspClassifierWrapper(env, grasp_reward_classifier_func=grasp_classifier_func)
    env = GraspClassifierNoVisionWrapper(env)

    if args.backward:
        env.set_task_id(1)
    else:
        env.set_task_id(0)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"replayed_bc_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)
    transitions = []
    trajectories = []
    num_noop_actions = 0
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")
    
    for i, trajectory in tqdm(enumerate(demo_data), desc="Replaying demos"):
        obs = trajectory['observations']
        action = trajectory['actions']
        next_obs = trajectory['next_observations']
        reward = trajectory['rewards']
        done = trajectory['dones']
        if done:
            env.grasp_queue = deque([False] * 5, maxlen=5)  # 최근 grasp 상태 저장
        else:
            reward = env.compute_grasp(next_obs, {})
        # if reward > 0.0:
        #     rew = reward
        # else:
        #     rew = env.env.reward_classifier_funcs[env.task_id](trajectory['next_observations']).item()
        #     rew = float(nn.sigmoid(rew))
        
        # if (rew >= 0.9):
        #     done = 1
        #     rew = 1.0
        # else:
        #     done = 0
        #     rew = 0.0
        # reward = env.compute_grasp(next_obs, {})
        # reward, done = env.relabel_obs(next_obs, {})
        
        scale_factor = 2

        # img = np.rot90(next_obs['front'], 2)
        img = next_obs['front']
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환
        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Agent View", frame_resized)

        # img = np.rot90(next_obs['wrist'], 2)
        img = next_obs['wrist']
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환
        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Robot View", frame_resized)

        cv2.waitKey(1)
        print(reward, done)
        new_transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=reward,
                masks=1-done,
                dones=done,
            )
        )
        transitions.append(new_transition)
    trajectories.append(transitions)

    with open(demo_save_path, "wb") as f:
        pkl.dump(trajectories, f)
        print(f"Saved {len(trajectories)} transitions to {demo_save_path}")

    env.close()
