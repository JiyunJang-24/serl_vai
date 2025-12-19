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
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper, SERLObsRobosuiteWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark
from libero.libero.envs import *
from libero.libero.envs import OffScreenRenderEnv
import argparse
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import matplotlib.pyplot as plt
import jax
import cv2
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
)
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
                
                img = data['next_observations']['agentview_image']
                img2 = data['next_observations']['robot0_eye_in_hand_image']
                obs = data['next_observations']
                reward = nn.sigmoid(classifier_func(obs).item())

                print(reward)
                frame = np.rot90(img, k=2)
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

# 사용 예시

options = {}

# print welcome info
print("Welcome to robosuite v{}!".format(suite.__version__))
print(suite.__logo__)

# Choose environment and add it to options
options["env_name"] = choose_environment()

# If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
if "TwoArm" in options["env_name"]:
    # Choose env config and add it to options
    options["env_configuration"] = choose_multi_arm_config()

    # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
    if options["env_configuration"] == "bimanual":
        options["robots"] = "Baxter"
    else:
        options["robots"] = []

        # Have user choose two robots
        print("A multiple single-arm configuration was chosen.\n")

        for i in range(2):
            print("Please choose Robot {}...\n".format(i))
            options["robots"].append(choose_robots(exclude_bimanual=True))

# Else, we simply choose a single (single-armed) robot to instantiate in the environment
else:
    options["robots"] = choose_robots(exclude_bimanual=True)

# Choose controller
controller_name = choose_controller()

# Load the desired controller
options["controller_configs"] = load_controller_config(default_controller=controller_name)
task_description = options['env_name'] + "_" + options['robots']
# initialize the task
env = suite.make(
    **options,
    has_renderer=True,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=True,
    control_freq=20,
    camera_names=["robot0_eye_in_hand", "agentview"],
    render_gpu_device_id=1,
)
env.reset()
env.viewer.set_camera(camera_id=0)

env = SERLObsRobosuiteWrapper(env)
env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
env = FrontCameraLIBEROWrapper(env)

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
    sample=env.image_observation_space.sample(),
    image_keys=image_keys,
    checkpoint_path="/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/classifier_ckpt/fw_checkpoint_1000",
)

# import pdb; pdb.set_trace()


folder_path = "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/bc_demos/fw_neg"  # 폴더 경로 입력
process_all_pickles(folder_path, classifier_func, env)
