import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite import load_controller_config

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark
from libero.libero.envs import *
from libero.libero.envs import OffScreenRenderEnv
import argparse
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import matplotlib.pyplot as plt
import cv2

def save_video(frames, filename, fps=30):
    """ 주어진 이미지 리스트(frames)를 MP4 동영상으로 저장 """
    if len(frames) == 0:
        print("No frames to save!")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 코덱
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR을 사용하므로 변환
        out.write(frame_bgr)

    out.release()
    print(f"Saved video: {filename}")

def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", type=str, required=True, help="Name of the demo file (e.g., bw_pick~.pkl)")
    parser.add_argument("--classifier", type=int, required=True, help="whether the demo is for classifier")
    args = parser.parse_args()

    # Define paths
    if args.classifier:
        demo_load_path = os.path.join("sc_demos_from_mac", args.demo_file)
        demo_save_path = os.path.join("sc_demos", args.demo_file)
    else:
        demo_load_path = os.path.join("demos_from_mac", args.demo_file)
        demo_save_path = os.path.join("demos", args.demo_file)
        initial_save_path = os.path.join("demos", "initial_state/"+args.demo_file)
    # Load demo data
    if not os.path.exists(demo_load_path):
        raise FileNotFoundError(f"Demo file not found: {demo_load_path}")

    with open(demo_load_path, "rb") as f:
        demo_data = pkl.load(f)

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

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"replayed_bc_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)
    transitions = []
    trajectories = []
    initial_states = []
    num_noop_actions = 0
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")
    video_frames = []  # 이미지 프레임 저장 리스트
    for trajectory in tqdm(demo_data, desc="Replaying demos"):
        init_state = trajectory[0]
        transitions = []
        env.env.env.env.sim.set_state_from_flattened(init_state)
        obs = env.env.observation_()
        for step in range(1, len(trajectory)):
            if step % 500 == 0:
                init_state = env.env.env.env.sim.get_state().flatten()
                env.reset()
                env.env.env.env.sim.set_state_from_flattened(init_state)
                obs = env.env.observation_()
            action = trajectory[step]["actions"]
            prev_action = transitions[-1]['actions'] if len(transitions) > 0 else None
            if is_noop(action, prev_action):
                print(f"\tSkipping no-op action: {action}")
                num_noop_actions += 1
                continue
            next_obs, rew, done, info = env.step(action=action)
            rew = 0
            done = 0
            new_transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1-done,
                    dones=done,
                )
            )
            transitions.append(new_transition)
            if obs['agentview_image'].shape[0] == 1:
                rotated_iamge = obs['agentview_image'].squeeze(0)
            else:
                rotated_iamge = obs['agentview_image']
            video_frames.append(rotated_iamge)
            obs = next_obs
            # if done == 1:
            #     rotated_image = np.rot90(obs['agentview_image'].squeeze(0), k=2)
            #     plt.imshow(rotated_image)
            #     plt.axis("off")
            #     plt.show()
            #     import pdb; pdb.set_trace()
           
        transitions[-1]["dones"] = 1
        transitions[-1]["masks"] = 0
        transitions[-1]["rewards"] = 1
        # rotated_image = np.rot90(obs['agentview_image'].squeeze(0), k=2)
        save_video(video_frames, "trajectory_video.mp4")
        print("Num noop action: ", num_noop_actions)
        # trajectories.append(transitions)
        if args.classifier:
            trajectories.append(transitions)
        else:
            rotated_image = np.rot90(obs['agentview_image'].squeeze(0), k=2)
            plt.imshow(rotated_image)
            plt.axis("off")
            plt.show() 
            save = input("Save trajectory or End? (y/n/e): ")
            if save == "n":
                continue
            elif save == "y":
                initial_states.append(trajectory[0])  
                trajectories.append(transitions)
            elif save == "e":
                break
        env.reset()

    with open(demo_save_path, "wb") as f:
        pkl.dump(trajectories, f)
        print(f"Saved {len(trajectories)} transitions to {demo_save_path}")

    if args.classifier == False:
        with open(initial_save_path, "wb") as f:
            pkl.dump(initial_states, f)
            print(f"Saved {len(initial_states)} transitions to {demo_save_path}")

    env.close()
