import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
# import keyboard
import pygame
# from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
# from robosuite import load_controller_config
# from robosuite.controllers import load_controller_config
# from robosuite.utils.input_utils import *

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, Quat2EulerWrapper, SERLObsRobosuiteWrapper, SERLObsRobosuiteInstructionWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, GripperPenaltyWrapper, GraspClassifierNoVisionWrapper
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

dummy_action = np.array([0.] * 7)

sub_task_list = [
    "pick up the bottle from the light brown bin",
    "place the bottle on the white desk"
    # "Pick up the cup from the light brown bin",
    # "Place the cup on the light brown bin",
]
pose_precise_control_list = [False, True]

if __name__ == "__main__":
    env = gym.make("box_picking_camera_env",
                   camera_mode="rgb",
                   max_episode_length=1000,
                   )
    env = SpacemouseInterventionUR5(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    # env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)
    env = GraspClassifierNoVisionWrapper(env)
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
    sub_task_id = 0

    pygame.init()
    pygame.display.set_mode((400, 400))  # 창을 띄우지 않으면 이벤트 큐가 초기화되지 않음

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force, 'reset_pose': env.unwrapped.curr_reset_pose}
 
    pbar = tqdm(total=success_needed, desc="demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    file_names = [f"{sub_task_list[i]}_{uuid}.pkl".replace(" ", "_").replace("-", "_") for i in range(len(sub_task_list))]

    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vla_demos")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    task_folder_path_list = []
    for i in range(len(sub_task_list)):
        task_folder_path = os.path.join(file_dir, sub_task_list[i].replace(" ", "_").replace("-", "_"))
        if not os.path.exists(task_folder_path):
            os.mkdir(task_folder_path)
        task_folder_path_list.append(task_folder_path)

    file_name_paths = [os.path.join(task_folder_path_list[i], file_names[i]) for i in range(len(sub_task_list))]

    # save trajectries by sub task
    sub_task_trajectory_dict = {i: [] for i in range(len(sub_task_list))}


    actions = np.zeros((7,))
    step = 0
    running_reward = 0.
    while success_count < success_needed:
        for i in range(len(sub_task_list)):
            if len(sub_task_trajectory_dict[i]) == 0:
                print(f"Current the number of task: {sub_task_list[i]} is 0")
            else:
                print(f"Current the number of task: {sub_task_list[i]} is {len(sub_task_trajectory_dict[i])}")

        task_id = input("Enter start (z) and save and exit (q): ")
        print("Save buttom: 's', Done button: 'd', Exit button: 'e', Next sub task: 'n'")
        if task_id == 'z':
            obs, _ = env.reset(task_id=0)
            task_forward = True
            positive = True
            obs['state']['target_orientation'] = np.array([0.0, 0.0, 0.0])
        
        elif task_id == 'q':
            break
        else:
            print("Invalid input")
            continue
        #transtions by the task
        transitions = []
        for i in range(len(sub_task_list)):
            transitions.append([])

        is_done = False
        is_save = False
        is_exit = False
        sub_task_id = 0
        while True:
            step += 1
            next_obs, rew, done, info = env.step(action=actions)
            print(next_obs['state']['tcp_pose'])
            next_obs['state']['target_orientation'] = np.array([0.0, 0.0, 0.0])
            rew = 0.0
            done = 0.0
            # import pdb; pdb.set_trace()
            if info['gripper_working'] is False:
                pause = True
                print("Gripper not working, pause the training")
            # print(next_obs['state']['tcp_pose'])
            if "intervene_action" in info:
                actions = info["intervene_action"]
            # print(actions)
            if is_save:
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
                transitions[sub_task_id].append(transition)
                
            obs = next_obs
            running_reward += rew
        
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        is_done = True
                        print("Done!!")
                    elif event.key == pygame.K_s:
                        is_save = True
                        print("Start saving transitions!!")
                    elif event.key == pygame.K_e:
                        is_exit = True
                        print("Exit!!")
                    elif event.key == pygame.K_n:
                        sub_task_id += 1
                        print(f"Sub task id: {sub_task_id}")
            if is_exit:
                print("Exiting...")
                break

            if is_done:
                rew = 1.0
                done = 1.0
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
                transitions[sub_task_id].append(transition)
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                # obs, _ = env.reset()
                print("Reward total:", running_reward)
                running_reward = 0.0
                for i, trans in enumerate(transitions):   
                    trans[-1]['rewards'] = 1.0
                    trans[-1]['masks'] = 0.0 
                    trans[-1]['dones'] = 1.0
                    if pose_precise_control_list[i]:
                        target_orientation = trans[-1]['observations']['state']['tcp_pose'][-3:]
                        for tran in trans:
                            tran['observations']['state']['target_orientation'] = target_orientation
                    sub_task_trajectory_dict[i] += trans
                rew = 0.0
                done = 0.0
                break

    for i in range(len(sub_task_list)):
        if len(sub_task_trajectory_dict[i]) != 0:
            with open(file_name_paths[i], "wb") as f:
                pkl.dump(sub_task_trajectory_dict[i], f)
                print(f"saved {len(sub_task_trajectory_dict[i])} transitions to {file_name_paths[i]}")
        else:
            print(f"No transitions saved for {sub_task_list[i]}")

    pbar.close()
    env.close()
