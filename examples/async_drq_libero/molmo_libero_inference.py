import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os


from robosuite import load_controller_config

from libero.libero.envs import *
import argparse
from libero.libero import benchmark
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=0.9,
        help="How much to scale rotation user inputs",
    )
    
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # Get controller config

    # Create argument configuration
    config = {
        "robots": args.robots,
    }
    
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    task_id = 0
    task = task_suite.get_task(task_id)
    # retrieve a specific task
    task_name = task.name
    task_description = task.language
    task_bddl_file = os.path.join("/home/fick17/Desktop/JY/SERL/serl/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")
    scale_factor = 2
    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256
    }
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    actions = np.zeros((7,))
    step = 0
    while 1:
        step += 1
        
        # actions = action_from_molmo(obs, task_description)
        next_obs, rew, done, info = env.step(action=actions)

        img = np.rot90(next_obs['agentview_image'], 2)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환
        
        frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
        cv2.imshow("Episode Video", frame_resized)
        cv2.waitKey(1)


        obs = next_obs
    
    env.close()
