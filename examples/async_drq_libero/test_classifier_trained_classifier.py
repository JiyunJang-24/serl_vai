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

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper
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


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_spatial"
task_suite = benchmark_dict[task_suite_name]()
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join("/home/fick17/Desktop/JY/SERL/serl/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")

env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id)
init_state_id = 0
env.set_init_state(init_states[init_state_id])
env = SERLObsLIBEROWrapper(env)
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
    checkpoint_path="/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/classifier_ckpt/grasp_checkpoint_1000",
)

# import pdb; pdb.set_trace()


folder_path = "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/bc_demos/grasp_pos"  # 폴더 경로 입력
process_all_pickles(folder_path, classifier_func, env)
