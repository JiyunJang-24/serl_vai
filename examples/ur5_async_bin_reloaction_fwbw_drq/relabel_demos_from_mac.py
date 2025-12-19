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
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierWrapper
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
    parser.add_argument("--sparse_reward", type=int, default=0, help="Use sparse reward")
    parser.add_argument("--grasp_reward", type=int, default=0, help="Use sparse reward")
    parser.add_argument("--backward", type=int, default=0, help="backward")
    parser.add_argument("--fw_reward_classifier_ckpt_path", type=str, default=None, required=True, help="Path to the forward reward classifier checkpoint")
    parser.add_argument("--bw_reward_classifier_ckpt_path", type=str, default=None, required=True, help="Path to the backward reward classifier checkpoint")
    parser.add_argument("--grasp_reward_classifier_ckpt_path", type=str, default=None, required=False, help="Path to the grasp reward classifier checkpoint")
    args = parser.parse_args()

    demo_load_path = os.path.join("demos", args.demo_file)
    demo_initial_path = os.path.join("demos", "initial_state/"+args.demo_file)
    demo_save_path = os.path.join("relabel_demos_dense_reward", args.demo_file)

    # Load demo data
    if not os.path.exists(demo_load_path):
        raise FileNotFoundError(f"Demo file not found: {demo_load_path}")
    
    if not os.path.exists(demo_initial_path):
        raise FileNotFoundError(f"Demo initial file not found: {demo_initial_path}")

    with open(demo_load_path, "rb") as f:
        demo_data = pkl.load(f)

    with open(demo_initial_path, "rb") as f:
        initial_states = pkl.load(f)

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
    env_image_observation_space = env.image_observation_space
    from serl_launcher.networks.reward_classifier import load_classifier_func

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    if (
        not args.fw_reward_classifier_ckpt_path
        or not args.bw_reward_classifier_ckpt_path
    ):
        raise ValueError(
            "Must provide both fw and bw reward classifier ckpt paths for actor"
        )
    fw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=front_image_keys,
        checkpoint_path=args.fw_reward_classifier_ckpt_path,
    )
    rng, key = jax.random.split(rng)
    bw_classifier_func = load_classifier_func(
        key=key,
        sample=env.front_observation_space.sample(),
        image_keys=front_image_keys,
        checkpoint_path=args.bw_reward_classifier_ckpt_path,
    )
    if args.sparse_reward:
        env = FWBWFrontCameraBinaryRewardClassifierWrapper(
            env, fw_classifier_func, bw_classifier_func
        )
    else:
        env = FWBWFrontCameraRewardClassifierWrapper(
            env, fw_classifier_func, bw_classifier_func
        )
    # rng, key = jax.random.split(rng)
    # grasp_classifier_func = load_classifier_func(
    #         key=key,
    #         sample=env_image_observation_space.sample(),
    #         image_keys=image_keys,
    #         checkpoint_path=args.grasp_reward_classifier_ckpt_path
    #     )
    # env = GraspClassifierWrapper(env, grasp_classifier_func)

    if args.backward:
        env.set_task_id(1)

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
        initial_state = initial_states[i]
        transitions = []
        env.env.env.env.env.env.env.sim.set_state_from_flattened(initial_state)
        obs = env.env.env.env.observation_()
        for step in range(1, len(trajectory)):
            if step % 500 == 0:
                initial_state = env.env.env.env.env.env.env.sim.get_state().flatten()
                env.reset()
                env.env.env.env.env.env.env.sim.set_state_from_flattened(initial_state)
                obs = env.env.env.env.env.observation_()
            action = trajectory[step]["actions"]
            prev_action = transitions[-1]['actions'] if len(transitions) > 0 else None
            if is_noop(action, prev_action):
                print(f"\tSkipping no-op action: {action}")
                num_noop_actions += 1
                continue
            next_obs, rew, done, info = env.step(action=action)
            
            print(rew, done)
            frame = np.rot90(next_obs['agentview_image'], k=2)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR 변환
            cv2.imshow("Thrid camera view", frame_bgr)

            img2 = next_obs['robot0_eye_in_hand_image']
            frame = np.rot90(img2, k=2)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # BGR 변환
            cv2.imshow("Hand camera view", frame_bgr)

            cv2.waitKey(1)  # 1ms 대기
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
            obs = next_obs
            # if done == 1:
            #     rotated_image = np.rot90(obs['agentview_image'].squeeze(0), k=2)
            #     plt.imshow(rotated_image)
            #     plt.axis("off")
            #     plt.show()
            #     import pdb; pdb.set_trace()
        # rotated_image = np.rot90(obs['agentview_image'].squeeze(0), k=2)
        # plt.imshow(rotated_image)
        # plt.axis("off")
        # plt.show()
                    
        print("Num noop action: ", num_noop_actions)
        trajectories.append(transitions)
        # save = input("Save trajectory? (y/n): ")
        # if save == "n":
        #     continue
        # elif save == "y":
        #     trajectories.append(transitions)
        env.reset()

    with open(demo_save_path, "wb") as f:
        pkl.dump(trajectories, f)
        print(f"Saved {len(trajectories)} transitions to {demo_save_path}")

    env.close()
