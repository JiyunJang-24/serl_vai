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
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, GripperPenaltyWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierWrapper, GraspClassifierNoVisionWrapper
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
# from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, ToMrpWrapper, ObservationRotationWrapper, Quat2EulerWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper

dummy_action = np.array([0.] * 7)


if __name__ == "__main__":
    env = gym.make("box_picking_camera_env",
                   camera_mode="rgb",
                   max_episode_length=200,
                   save_video=True,
                #    only_pos_control=False
                   )
    env = SpacemouseInterventionUR5(env)
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
    # grasp_reward_classifier_ckpt_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/grasp_checkpoint_30000"
    # rng = jax.random.PRNGKey(0)
    # rng, key = jax.random.split(rng)
    # grasp_classifier_func = load_classifier_func(
    #         key=key,
    #         sample=env.wrist_observation_space.sample(),
    #         image_keys=wrist_image_keys,
    #         checkpoint_path=grasp_reward_classifier_ckpt_path
    #     )
    env = GraspClassifierNoVisionWrapper(env)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    obs, _ = env.reset()
    print("Initial reset done")
    transitions = []
    success_count = 0
    success_needed = 40
    total_count = 0
    pbar = tqdm(total=success_needed)
    is_save = False
    is_done = False
    is_exit = False
    forward_pos_trajectories = []
    forward_neg_trajectories = []

    def on_press(key):
        global is_save
        global is_exit
        if key == keyboard.Key.f1 and not is_save:
            is_save = True
            print("Save Start !!!!!!!!!")
        elif key == keyboard.Key.f2:
            is_exit = True
            print("Exit !!!!!!!!!")

    
    def on_esc(key):
        global is_done
        if key == keyboard.Key.esc and not is_done:
            is_done = True
            print("Done !!!!!!!!!")

 

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force, 'reset_pose': env.unwrapped.curr_reset_pose}
 

    pbar = tqdm(total=success_needed, desc="demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    fw_pos_file_name = f"grasp__demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")

    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "grasp_demos")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    fw_pos_file_path = os.path.join(file_dir, fw_pos_file_name)

    actions = np.zeros((7,))
    step = 0
    running_reward = 0.
    forward_count = 0

    while success_count < success_needed:
        print("Current the number of task: forward {}".format(
            len(forward_pos_trajectories)
        ))
        print("Current the number of task demo: forward {}".format(
            forward_count,
            
        ))
        task_id = input("Enter forward (z), save and exit (q): ")
        if task_id == 'z':
            obs, _ = env.reset(task_id=0)
            task_forward = True
            positive = True
        
        elif task_id == 'q':
            break
        else:
            print("Invalid input")
            continue
        transitions = []
        while True:
            step += 1
            next_obs, rew, done, info = env.step(action=actions)
            print(rew, next_obs['state']['tcp_pose'][:3])
            if rew == 0.3 and next_obs['state']['tcp_pose'][2] < -0.06 and next_obs['state']['tcp_pose'][0] < 0.02 and next_obs['state']['tcp_pose'][0] > -0.02 and next_obs['state']['tcp_pose'][1] < 0.02 and next_obs['state']['tcp_pose'][1] > -0.02:
                rew = 1.0
                done = True
            else:
                done = False
            # print(next_obs['state'][5:11])
            if "intervene_action" in info:
                actions = info["intervene_action"]
            # print(actions)
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)

            obs = next_obs
            running_reward += rew
            # print(rew)
            if done:
                success_count += int(rew > 0.9)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.9))
                # obs, _ = env.reset()
                print("Reward total:", running_reward)
                running_reward = 0.0
                if int(rew > 0.9):
                    while True:
                        is_save = input("Do you want to save the current trajectory? (y/n)")
                        if is_save.lower() == 'y':
                            if rew > 0.9:
                                if task_forward:
                                    forward_pos_trajectories += transitions
                                    forward_count += 1  
                            break
                        elif is_save.lower() == 'n':
                            print("Not saving the current trajectory.")
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                            continue                    
                else:
                    print("Fail!!!")
                transitions = []
                break

    if len(forward_pos_trajectories) != 0:
        with open(fw_pos_file_path, "wb") as f:
            pkl.dump(forward_pos_trajectories, f)
            print(f"saved {len(forward_pos_trajectories)} transitions to {fw_pos_file_path}")
  

    pbar.close()
    env.close()
    # listener_1.stop()
    # listener_2.stop()
