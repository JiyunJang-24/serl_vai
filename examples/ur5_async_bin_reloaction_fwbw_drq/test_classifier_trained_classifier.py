import pickle
import numpy as np
import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import flax.linen as nn
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite import load_controller_config
import jax.numpy as jnp
import jax

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, Quat2EulerWrapper, SERLObsRobosuiteWrapper, SERLObsRobosuiteInstructionWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, GripperPenaltyWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO, SpacemouseInterventionUR5
# import libero.libero.envs.bddl_utils as BDDLUtils
# from libero.libero.envs import *
import argparse
# from libero.libero import benchmark
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import cv2
import ur_env
import threading
import keyboard
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, ToMrpWrapper, ObservationRotationWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper
sigmoid = lambda x: 1 / (1 + np.exp(-x))

import numpy as np
from flax.core import frozen_dict
from typing import Dict, Any, Optional


def analyze_pickle(file_path, classifier_func, env):
    """Load and analyze the contents of a pickle file."""
    try:
        scale_factor = 2  # 화면 크기 조절
        
        with open(file_path, "rb") as f:
            data_list = pickle.load(f)
            for data in data_list:
                
                img = data['next_observations']['front']
                img2 = data['next_observations']['wrist']
                obs = data['next_observations']
                reward = nn.sigmoid(classifier_func(obs).item())

                print(reward)
                # frame = np.rot90(img, k=2)
                frame = img
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR 변환

                frame_resized = cv2.resize(
                    frame_bgr, 
                    (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor)
                )
                cv2.imshow("Agent View", frame_resized)
                frame = np.rot90(img2, k=2)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR 변환

                frame_resized = cv2.resize(
                    frame_bgr, 
                    (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor)
                )
                cv2.imshow("Hand View", frame_resized)
                cv2.waitKey(1)

    except Exception as e:
        print(f"Error loading pickle file ({file_path}): {e}")

def process_all_pickles(folder_path, classifier_func, env):
    """Find all pickle files in a folder and process them."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    pickle_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    if not pickle_files:
        print(f"No pickle files found in {folder_path}")
        return

    for pkl_file in sorted(pickle_files):  # 정렬된 순서로 파일 처리
        full_path = os.path.join(folder_path, pkl_file)
        print(f"Processing: {full_path}")
        analyze_pickle(full_path, classifier_func, env)


if __name__ == "__main__":
    env = gym.make("box_picking_camera_env",
                    camera_mode="rgb",
                    max_episode_length=200,
                    fake_env=True
                    )
    env = SpacemouseInterventionUR5(env, fake_env=True)
    env = RelativeFrame(env)
    env = ToMrpWrapper(env)
    env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    front_image_keys = [
        k for k in env.front_observation_space.keys() if "state" not in k
    ]

    from serl_launcher.networks.reward_classifier import load_classifier_func
    # import pdb; pdb.set_trace
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=front_image_keys,
        checkpoint_path="/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/fw_checkpoint_1000",
    )

    # import pdb; pdb.set_trace()


    folder_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/bc_demos/fw_neg"  # 폴더 경로 입력
    process_all_pickles(folder_path, classifier_func, env)
