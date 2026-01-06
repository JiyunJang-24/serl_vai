"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import sys, termios, tty
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
import gym
import pygame
import draccus
import multiprocessing as mp
import torch
import wandb
# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
import json
from experiments.robot.ur5.ur5_utils import (
    get_next_task_label,
    get_preprocessed_image,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
    ur5_to_openvla_obs
)
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
import numpy as np
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsSubGoalWrapper, SERLObsLIBEROWrapper, SERLObsRobosuiteWrapper, RelativeFrame, Quat2EulerWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierNoVisionWrapper, GraspClassifierWrapper, GraspClassifierRobosuiteWrapper, GripperPenaltyWrapper, GripperPenaltyUR5Wrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO, SpacemouseInterventionUR5
import ur_env
IMAGE_RESOLUTION = 256
lerobot_root = "/home/vai/Desktop/yujin/shortcut-learning-in-grps/lerobot"
if lerobot_root not in sys.path:
    sys.path.append(lerobot_root)
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

id_to_task = {#0: "pick_up_the_brown_cat_doll_from_the_light_wood_bin_and_place_the_cat_doll_in_the_dark_wood_bin",
                0: "Pick up the gray plush from the bin and place it on the brown plate", 
            #    1: "pick_up_the_brown_cat_doll_from_the_dark_wood_bin_and_place_the_cat_doll_in_the_light_wood_bin",
            #    2: "pick_up_the_purple_doll_from_the_light_wood_bin_and_place_the_purple_doll_on_the_tape",
            #    3: "pick_up_the_purple_doll_from_the_light_wood_bin_and_place_the_purple_doll_in_the_dark_wood_bin"
               }

def _json_to_tensors(obj):
    """stats.json을 로드했을 때 list/number들을 torch.Tensor로 재귀 변환"""
    if isinstance(obj, dict):
        return {k: _json_to_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # 리스트는 float32 텐서로
        return torch.tensor(obj, dtype=torch.float32)
    if isinstance(obj, (int, float, np.integer, np.floating)):
        # 스칼라는 텐서로 (정말 스칼라여야 할 곳이 많음)
        return torch.tensor(obj, dtype=torch.float32)
    # 이미 텐서면 그대로
    if isinstance(obj, torch.Tensor):
        return obj
    return obj  # 그 외 타입은 그대로 (예: 문자열 키 등)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "diffusion"                               # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/vai/Desktop/yujin/shortcut-learning-in-grps/lerobot/outputs/train/2026-01-05/01-37-46_DP_reward_debug/checkpoints/050000/pretrained_model"   # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 75                                         # Max number of timesteps per episode
    control_frequency: float = 10                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    use_wandb:  bool = False                                    # Whether to use Weights & Biases logging
    # fmt: on


@draccus.wrap()
def eval_model_in_ur5_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"
    # Initialize local logging


    # TODO : 수정해야함
    # # [OpenVLA] Set action un-normalization key
    # cfg.unnorm_key = "vai_ur5e_tiger_on_tape_openvla_type"
    if cfg.model_family == "diffusion":
        pretrained_policy_path = cfg.pretrained_checkpoint
        stats_path = pretrained_policy_path+"/stats.json"
        
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_raw = json.load(f)

        dataset_stats = _json_to_tensors(stats_raw)
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, dataset_stats=dataset_stats, evaluation=True)


    # Get expected image dimensions
    if cfg.model_family == "diffusion":
        resize_size = get_image_resize_size(cfg) if IMAGE_RESOLUTION is None else IMAGE_RESOLUTION

    pygame.init()
    pygame.display.set_mode((800, 800))  # 창을 띄우지 않으면 이벤트 큐가 초기화되지 않음
    # Initialize the WidowX environment
    env = gym.make("box_picking_camera_env",
                    camera_mode="rgb",
                    max_episode_length=cfg.max_steps,
                    only_pos_control=True,
                    )
    # env = SpacemouseInterventionUR5(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    # env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    # env = SERLObsSubGoalWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)
    env = GraspClassifierNoVisionWrapper(env)
    

    # Start evaluation
    task_label = ""
    episode_idx = 0
    # env.set_task_id(1)
    task_id = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        # task_label = get_next_task_label(task_label)
        task_name = id_to_task[0]
        task_name = " ".join(task_name.split("_"))
        
        # Reset environment
        obs, _ = env.reset(task_id=env.task_id)
        is_not_done = True
        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []
            
        if cfg.model_family == "diffusion":
            policy.reset()
        # Start episode
        #input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        is_not_done = False
                        task_id = (task_id + 1) % 2
                        print("Success!!")
            try:
                if is_not_done:
                    curr_tstamp = time.time()
                    if curr_tstamp > last_tstamp + step_duration:

                        print(f"t: {t}")
                        print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                        last_tstamp = time.time()
                        # Refresh the camera image and proprioceptive state
                        obs, state = refresh_obs(obs, env)
                        obs = ur5_to_openvla_obs(obs, env)
                        
                        # Save full (not preprocessed) image for replay video
                        replay_images.append(obs["full_image"])

                        # Get preprocessed image
                        obs["full_image"] = get_preprocessed_image(obs, resize_size)


                        if cfg.model_family == "diffusion":

                            state = torch.from_numpy(state["tip_pose"])
                            image = torch.from_numpy(obs["full_image"].copy())
                            state = state.to(torch.float32)
                            image = image.to(torch.float32) / 255.0
                            image = image.permute(2, 0, 1)

                            # Send data tensors from CPU to GPU
                            state = state.to('cuda', non_blocking=True)
                            image = image.to('cuda', non_blocking=True)

                            # Add extra (empty) batch dimension, required to forward the policy
                            state = state.unsqueeze(0)
                            image = image.unsqueeze(0)
                            
                            observation = {
                                "observation.state": state,
                                "observation.image": image,
                                "task": task_name,
                            }
                            with torch.inference_mode():
                                action = policy.select_action(observation)
                            action = action.squeeze(0).to("cpu").numpy()
                            action = action[[0,1,2,6]]

                        # [If saving rollout data] Save preprocessed image, robot state, and action
                        if cfg.save_data:
                            rollout_images.append(obs["full_image"])
                            rollout_actions.append(action)

                        # Execute action
                        print("action:", action)

                        obs, reward, done, info = env.step(action)
                        t += 1
            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break
        
        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    eval_model_in_ur5_env()
