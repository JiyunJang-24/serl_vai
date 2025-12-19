import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import robosuite

# from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
# from robosuite import load_controller_config
# from robosuite.controllers import load_controller_config
# from robosuite.utils.input_utils import *

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, Quat2EulerWrapper, SERLObsRobosuiteWrapper, SERLObsRobosuiteInstructionWrapper, RelativeFrame
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, GripperPenaltyWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO, SpacemouseInterventionUR5
# import libero.libero.envs.bddl_utils as BDDLUtils
# from libero.libero.envs import *
import argparse
# from libero.libero import benchmark
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import cv2
import ur_env
import threading
import jax
import keyboard
import pygame
# from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, ToMrpWrapper, ObservationRotationWrapper, Quat2EulerWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper

dummy_action = np.array([0.] * 7)


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_mode((800, 800))  # 창을 띄우지 않으면 이벤트 큐가 초기화되지 않음
    pygame.display.set_caption("Press f for fw, b for bw")  # 창 이름(제목) 설정
    env = gym.make("box_picking_camera_env",
                   camera_mode="rgb",
                   max_episode_length=200,
                   save_video=True
                   )
    env = SpacemouseInterventionUR5(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = ScaleObservationWrapper(env)
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
    grasp_classifier_func = load_classifier_func(
        key=key,
        sample=env.wrist_observation_space.sample(),
        image_keys=wrist_image_keys,
        checkpoint_path=grasp_classifier_ckpt_path,
    )
    env = GraspClassifierWrapper(env, grasp_reward_classifier_func=grasp_classifier_func)

    env.set_task_id(0)
    obs, _ = env.reset()

    while True:
        actions = np.zeros((7,))
        next_obs, rew, done, info = env.step(action=actions)
        if "real_reward" in info:
            print(info['real_reward'])

        # if 'grasp_prob' in info:
        #     print(info['grasp_prob'])
            

        if "intervene_action" in info:
            actions = info["intervene_action"]

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    env.reset(task_id=0)
                    print("Reset for Forward task !!")
                elif event.key == pygame.K_b:
                    env.reset(task_id=1)
                    print("Reset for Backward task !!")
        