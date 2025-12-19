import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import flax.linen as nn

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper, SERLObsRobosuiteWrapper
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


def is_grasping_bread(robot_pos, gripper_qpos, bread_pos, tcp_thresh=0.032, xy_thresh=0.035, height_thresh=0.32, gripper_close_thresh=0.06):
    """
    로봇이 빵을 잡았는지 여부를 반환.
    
    Args:
        tcp_pose: (7,) array. End-effector의 pose (x, y, z, qx, qy, qz, qw)
        gripper_qpos: (2,) array. 그리퍼의 관절 위치
        bread_pos: (3,) array. 빵의 중심 좌표
        tcp_thresh: 빵 중심과 tcp 사이의 허용 거리
        height_thresh: 그리퍼가 빵 위에 어느 정도 떠 있어야 함
        gripper_close_thresh: 그리퍼가 어느 정도 닫혀 있어야 함
    """
    tcp_pose = robot_pos
    gripper_qpos = gripper_qpos
    tcp_pos = tcp_pose[:3]
    tcp_bread_dist = np.linalg.norm(tcp_pos - bread_pos)

                # xy 평면 거리 차이 (개별 축 기준)
    xy_dist = np.linalg.norm(tcp_pos[:2] - bread_pos[:2])
    is_xy_near = xy_dist < xy_thresh
    
    is_near = tcp_bread_dist < tcp_thresh and is_xy_near
    is_above = bread_pos[2] > 0.85 and tcp_pose[2] > 0.85 and (tcp_pos[2] - bread_pos[2]) < height_thresh
    
    is_gripper_closed = (np.abs(gripper_qpos[0] - gripper_qpos[1]) <= gripper_close_thresh) and (np.abs(gripper_qpos[0] - gripper_qpos[1]) >= 0.0373)
    # print("is near: ", xy_dist, is_near, "is above:", (tcp_pos[2] - bread_pos[2]), is_above, "is gripper:", (np.abs(gripper_qpos[0] - gripper_qpos[1]), is_gripper_closed))
    return is_near and is_gripper_closed


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
    # demo_initial_path = os.path.join("demos", "initial_state/"+args.demo_file)
    demo_save_path = os.path.join("relabel_demos_dense_reward", args.demo_file)

    # Load demo data
    if not os.path.exists(demo_load_path):
        raise FileNotFoundError(f"Demo file not found: {demo_load_path}")
    
    # if not os.path.exists(demo_initial_path):
    #     raise FileNotFoundError(f"Demo initial file not found: {demo_initial_path}")

    with open(demo_load_path, "rb") as f:
        demo_data = pkl.load(f)

    # with open(demo_initial_path, "rb") as f:
    #     initial_states = pkl.load(f)

    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite_name = "libero_spatial"
    # task_suite = benchmark_dict[task_suite_name]()
    # task_id = 0
    # task = task_suite.get_task(task_id)
    # task_name = task.name
    # task_description = task.language
    # task_bddl_file = os.path.join("/home/fick17/Desktop/JY/SERL/serl/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
    
    # env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
    # env = OffScreenRenderEnv(**env_args)
    # env.seed(0)
    # env.reset()
    # init_states = task_suite.get_task_init_states(task_id)
    # init_state_id = 0
    # env.set_init_state(init_states[init_state_id])


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
        camera_names=["robot0_eye_in_hand", "robot0_eye_in_hand_left", "agentview"],
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

    done_check = False

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
        obs = trajectory['observations']
        action = trajectory['actions']
        next_obs = trajectory['next_observations']
        rew = trajectory['rewards']
        done = trajectory['dones']
        grasp_penalty = trajectory['grasp_penalty']
        bread_pos=trajectory['bread_pos']
        next_bread_pos=trajectory['next_bread_pos']
        obs_robot_pos = trajectory['obs_robot_pos']
        next_obs_robot_pos = trajectory['next_obs_robot_pos']
        # bread_pos = trajectory['Bread_pos']
        # rew = env.reward_classifier_funcs[env.task_id](trajectory['next_observations']).item()
        # rew = float(nn.sigmoid(rew))
        info = {}
        bread_pos_z = next_bread_pos[2]
        current_grasp_info = next_obs['state']['robot0_gripper_qpos'][0]
        act = action[-1]
        is_already_open = current_grasp_info > 0.036  # 완전히 열린 상태 기준 (0.04 근처)
        is_already_closed = current_grasp_info < 0.036  # 완전히 닫힌 상태 기준 (0.0 근처)

        if (act < -0.5 and is_already_closed) or (act > 0.5 and is_already_open):
            info["grasp_penalty"] = -0.15
        else:
            info["grasp_penalty"] = 0.0

        if is_grasping_bread(next_obs_robot_pos, obs['state']['robot0_gripper_qpos'], next_bread_pos):
            grasp_success = 1
            grasp_reward = 0.5
            if args.backward:
                penalty = -np.abs(next_obs_robot_pos[1] - (-0.07))
                penalty_bread = -np.abs(next_bread_pos[1] - (-0.07))
            else:
                penalty = -np.abs(next_obs_robot_pos[1] - 0.07)
                penalty_bread = -np.abs(next_bread_pos[1] - 0.07)

            penalty += -np.abs(next_obs_robot_pos[0] - 0.095)
            penalty += np.abs(np.clip(next_obs_robot_pos[2], 0, 0.88) - 0.80)*3
            grasp_reward += penalty
            grasp_reward += penalty_bread
        else:
            grasp_success = 0
            grasp_reward = -np.linalg.norm(next_bread_pos - next_obs_robot_pos[:3]) * 5.0

        if args.backward:
            success_box = np.array([[0.005, -0.14, 0.81], [0.19, -0.03, 0.90]])
            success_check = (success_box[0, 0] < next_bread_pos[0] < success_box[1, 0]) and (success_box[0, 1] < next_bread_pos[1] < success_box[1, 1]) and (success_box[0, 2] < next_bread_pos[2] < success_box[1, 2])
        else:
            success_box = np.array([[0.005, 0.05, 0.81], [0.19, 0.14, 0.90]])
            success_check = (success_box[0, 0] < next_bread_pos[0] < success_box[1, 0]) and (success_box[0, 1] < next_bread_pos[1] < success_box[1, 1]) and (success_box[0, 2] < next_bread_pos[2] < success_box[1, 2])
        

        # if success_check and done_check:
        #     print("Continue")
        #     continue

        # if success_check:
        #     done_check=True
        # else:
        #     done_check=False
        

        if success_check:
            rew = 1.0
            done = 1.0
        else:
            rew = 0.0
            done = 0.0
            rew = rew + grasp_reward

        
        # if (rew >= 0.9):
        #     done = 1
        #     rew = 1.0
        # else:
        #     done = 0
        #     rew = 0.0

        scale_factor = 2

        img = np.rot90(next_obs['agentview_image'], 2)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Agent View", frame_resized)

        img = np.rot90(next_obs['robot0_eye_in_hand_image'], 2)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Robot View", frame_resized)

        img = np.rot90(next_obs['robot0_eye_in_hand_left_image'], 2)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Robot 2 View", frame_resized)
        cv2.waitKey(1)
        print(rew, done)
        new_transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=rew,
                masks=1-done,
                dones=done,
                grasp_penalty=grasp_penalty,
                grasp_success=grasp_success,
                bread_pos=bread_pos,
                next_bread_pos=next_bread_pos,
                obs_robot_pos=obs_robot_pos,
                next_obs_robot_pos=next_obs_robot_pos,

            )
        )
        transitions.append(new_transition)
    trajectories.append(transitions)

    with open(demo_save_path, "wb") as f:
        pkl.dump(trajectories, f)
        print(f"Saved {len(trajectories)} transitions to {demo_save_path}")

    env.close()
