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

dummy_action = np.array([0.] * 7)


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
    backward_pos_trajectories = [] 
    backward_neg_trajectories = []

    pygame.init()
    pygame.display.set_mode((400, 400))  # 창을 띄우지 않으면 이벤트 큐가 초기화되지 않음

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

    
    # listener_1 = keyboard._listener(on_press=on_press)
    # listener_1.start()

    # listener_2 = keyboard._listener(on_press=on_esc)
    # listener_2.start()

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force, 'reset_pose': env.unwrapped.curr_reset_pose}
    # listener_1 = keyboard._listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    # listener_1.start()

    # listener_2 = keyboard._listener(on_press=on_esc, daemon=True)
    # listener_2.start()

    pbar = tqdm(total=success_needed, desc="demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    
    fw_pos_file_name = f"fw_pos_bc_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    bw_pos_file_name = f"bw_pos_bc_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    fw_neg_file_name = f"fw_neg_bc_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    bw_neg_file_name = f"bw_neg_bc_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")

    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bc_demos")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    fw_pos_file_path = os.path.join(file_dir, 'fw_pos/'+fw_pos_file_name)
    bw_pos_file_path = os.path.join(file_dir, 'bw_pos/'+bw_pos_file_name)
    fw_neg_file_path = os.path.join(file_dir, 'fw_neg/'+fw_neg_file_name)
    bw_neg_file_path = os.path.join(file_dir, 'bw_neg/'+bw_neg_file_name)

    actions = np.zeros((7,))
    step = 0
    running_reward = 0.
    while success_count < success_needed:
        print("Current the number of task: forward pos {}, forward neg {}, backward pos {} , backward neg {}".format(
            len(forward_pos_trajectories), len(forward_neg_trajectories),
            len(backward_pos_trajectories), len(backward_neg_trajectories)
        ))

        task_id = input("Enter forward pos (z), forward neg (x), backward pos (c), backward neg (v), save and exit (q): ")
        print("Save buttom: 's', Done button: 'd', Exit button: 'e'")
        if task_id == 'z':
            obs, _ = env.reset(task_id=0)
            task_forward = True
            positive = True

        elif task_id == 'x':
            obs, _ = env.reset(task_id=0)
            task_forward = True
            positive = False

        elif task_id == 'c':
            obs, _ = env.reset(task_id=1)
            task_forward = False
            positive = True

        elif task_id == 'v':
            obs, _ = env.reset(task_id=1)
            task_forward = False
            positive = False
        
        elif task_id == 'q':
            break
        else:
            print("Invalid input")
            continue
        transitions = []
        is_done = False
        is_save = False
        is_exit = False
        while True:
            step += 1
            next_obs, rew, done, info = env.step(action=actions)
            rew = 0.0
            done = 0.0
            # import pdb; pdb.set_trace()
            if info['gripper_working'] is False:
                pause = True
                print("Gripper not working, pause the training")
            # print(next_obs['state']['tcp_pose'])
            if "intervene_action" in info:
                actions = info["intervene_action"]
            print(actions)
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
                transitions.append(transition)
                
            obs = next_obs
            running_reward += rew
            # if keyboard.is_pressed('e'):
            #     is_done = True
            #     print("Done by 'e' key pressed!")

            # elif keyboard.is_pressed('s'):
            #     is_save = True
            #     print("Save by 's' key pressed!")
            # keyboard.on_press(on_press)
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
                transitions.append(transition)
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                # obs, _ = env.reset()
                print("Reward total:", running_reward)
                running_reward = 0.0
                if task_forward:
                    if positive:
                        forward_pos_trajectories += transitions
                    else:
                        forward_neg_trajectories += transitions
                else:
                    if positive:
                        backward_pos_trajectories += transitions
                    else:
                        backward_neg_trajectories += transitions
                transitions = []
                rew = 0.0
                done = 0.0
                break

    if len(forward_pos_trajectories) != 0:
        with open(fw_pos_file_path, "wb") as f:
            pkl.dump(forward_pos_trajectories, f)
            print(f"saved {len(forward_pos_trajectories)} transitions to {fw_pos_file_path}")

    if len(forward_neg_trajectories) != 0:
        with open(fw_neg_file_path, "wb") as f:
            pkl.dump(forward_neg_trajectories, f)
            print(f"saved {len(forward_neg_trajectories)} transitions to {fw_neg_file_path}")

    
    if len(backward_pos_trajectories) != 0:
        with open(bw_pos_file_path, "wb") as f:
            pkl.dump(backward_pos_trajectories, f)
            print(f"saved {len(backward_pos_trajectories)} transitions to {bw_pos_file_path}")    

    if len(backward_neg_trajectories) != 0:
        with open(bw_neg_file_path, "wb") as f:
            pkl.dump(backward_neg_trajectories, f)
            print(f"saved {len(backward_neg_trajectories)} transitions to {bw_neg_file_path}")

    pbar.close()
    env.close()
    # listener_1.stop()
    # listener_2.stop()
