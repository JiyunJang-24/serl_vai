import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os


from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite import load_controller_config
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, Quat2EulerWrapper, SERLObsRobosuiteWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
import argparse
from libero.libero import benchmark
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import cv2

xyz_bounding_box = np.array([[-0.12, -0.5, 0.8], [0.32, 0.55, 1.1]])
dummy_action = np.array([0.] * 7)
def clip_safety_box(obs, action):
    """Clip the pose to be within the safety box while allowing movement back inside."""
    
    # 현재 엔드 이펙터 위치
    current_pos = obs['state']['tcp_pose'][:3]

    # 액션을 적용한 후 예상되는 위치
    next_pos = current_pos + (action[:3] * 0.05)

    # Safety Box 범위 (예: xyz_bounding_box = (min_bounds, max_bounds))
    min_bounds, max_bounds = xyz_bounding_box  # 예: (np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max]))

    # Safety Box를 벗어난 조건
    out_of_bounds_low = next_pos < min_bounds
    out_of_bounds_high = next_pos > max_bounds

    # 들어오는(안쪽으로 향하는) 액션인지 확인
    moving_inward = ((action[:3] > 0) & out_of_bounds_low) | ((action[:3] < 0) & out_of_bounds_high)
    
    # dummy_action은 클리핑할 기본 템플릿 (action과 같은 shape)
    clipped_action = copy.deepcopy(dummy_action)
    clipped_action[3:] = action[3:]
    
    # Safety Box를 벗어나려 하고, 안으로 들어오는 방향이 아니라면 이동 제한
    clipped_action[:3] = np.where(out_of_bounds_low & ~moving_inward, 0, action[:3])
    clipped_action[:3] = np.where(out_of_bounds_high & ~moving_inward, 0, clipped_action[:3])

    return clipped_action


def check_impossible(env):
    obs = env.env.env.env.env._get_observations()
    
    # 객체들의 위치 정보를 리스트로 저장
    obj_pos_list = [
        obs['akita_black_bowl_1_pos'],
        obs['akita_black_bowl_2_pos'],
        obs['cookies_1_pos'],
        obs['glazed_rim_porcelain_ramekin_1_pos'],
        obs['plate_1_pos']
    ]
    
    # 불가능한 상황 판단
    for obj_pos in obj_pos_list:
        if (obj_pos[0] < xyz_bounding_box[0][0] or obj_pos[0] > xyz_bounding_box[1][0] or
            obj_pos[1] < xyz_bounding_box[0][1] or obj_pos[1] > xyz_bounding_box[1][1] or
            obj_pos[2] < xyz_bounding_box[0][2] or obj_pos[2] > xyz_bounding_box[1][2]):
            print(f"불가능한 상태 감지: {obj_pos}")  # 디버깅 출력
            return True
    return False

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "--robots",
    #     nargs="+",
    #     type=str,
    #     default="Panda",
    #     help="Which robot(s) to use in the env",
    # )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="single-arm-opposed",
    #     help="Specified environment configuration if necessary",
    # )
    # parser.add_argument(
    #     "--arm",
    #     type=str,
    #     default="right",
    #     help="Which arm to control (eg bimanual) 'right' or 'left'",
    # )
    # parser.add_argument(
    #     "--camera",
    #     type=str,
    #     default="agentview",
    #     help="Which camera to use for collecting demos",
    # )
    # parser.add_argument(
    #     "--controller",
    #     type=str,
    #     default="OSC_POSE",
    #     help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    # )
    # parser.add_argument("--device", type=str, default="spacemouse")
    # parser.add_argument(
    #     "--pos-sensitivity",
    #     type=float,
    #     default=1.0,
    #     help="How much to scale position user inputs",
    # )
    # parser.add_argument(
    #     "--rot-sensitivity",
    #     type=float,
    #     default=0.9,
    #     help="How much to scale rotation user inputs",
    # )
    
    # parser.add_argument("--vendor-id", type=int, default=9583)
    # parser.add_argument("--product-id", type=int, default=50734)

    # args = parser.parse_args()

    # # Get controller config
    # controller_config = load_controller_config(default_controller=args.controller)

    # # Create argument configuration
    # config = {
    #     "robots": args.robots,
    #     "controller_configs": controller_config,
    # }
    
    # from pynput import keyboard
    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    # task_suite = benchmark_dict[task_suite_name]()
    # task_id = 0
    # task = task_suite.get_task(task_id)
    # # retrieve a specific task
    # task_name = task.name
    # task_description = task.language
    # task_bddl_file = os.path.join("/home/fick17/Desktop/JY/SERL/serl/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl")
    # print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
    #     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # # step over the environment
    # env_args = {
    #     "bddl_file_name": task_bddl_file,
    #     "camera_heights": 256,
    #     "camera_widths": 256
    # }
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
    # env = OffScreenRenderEnv(**env_args)
    # env = VisualizationWrapper(env)
    # env = SERLObsVisualizationLIBEROWrapper(env)
    env = SERLObsRobosuiteWrapper(env)
    # env = RelativeFrame(env)
    # env = Quat2EulerWrapper(env)
    
    # env = SERLObsVisualizationLIBEROWrapper(env)
    env = SpacemouseInterventionLIBERO(env)

    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraLIBEROWrapper(env)
    obs, _ = env.reset()
    scale_factor = 2
    next_obs1, rew1, done1, info1 = env.step(action=np.zeros((7,)))
    demos_count = 0
    demos_needed = 20

    # 이미지 크기 키우기 (2배 확대)
    is_save = False
    is_done = False
    is_exit = False
    forward_pos_trajectories = []
    forward_neg_trajectories = []
    backward_pos_trajectories = [] 
    backward_neg_trajectories = []

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

    # Collect all event until released
    listener_1 = keyboard.Listener(on_press=on_press)
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc)
    listener_2.start()

    pbar = tqdm(total=demos_needed, desc="bc_demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fw_pos_file_name = f"grasp_pos_{task_description}_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    fw_neg_file_name = f"grasp_neg_{task_description}_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")

    fw_pos_file_name = f"grasp_pos__demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    fw_neg_file_name = f"grasp_neg__demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")

    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bc_demos")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    fw_pos_file_path = os.path.join(file_dir, 'grasp_pos/'+fw_pos_file_name)
    fw_neg_file_path = os.path.join(file_dir, 'grasp_neg/'+fw_neg_file_name)


    transitions = []
    actions = np.zeros((7,))
    # actions[0] = 0.1
    while demos_count < demos_needed:
        print("Current the number of task: grasp pos {}, grasp neg {},".format(
            len(forward_pos_trajectories), len(forward_neg_trajectories)
        ))
        task_id = input("Enter grasp positive(z), grasp negative(x), save and exit (q): ")

        print("Save buttom: F1, Done button: ESC, Exit button: F2")
        if task_id == 'z':
            task_forward = True
            positive = True
        elif task_id == 'x':
            task_forward = True
            positive = False
        elif task_id == 'q':
            break
        else:
            print("Invalid input")
            continue
        step = 0
        while True:
            step += 1
            if step > 900:
                print("step over 900, env reset")
                last_state = env.env.env.env.env.sim.get_state().flatten()
                obs, _ = env.reset()
                env.env.env.env.env.sim.set_state_from_flattened(last_state)
                obs = env.env.env.observation_()
                step = 0
            # actions = clip_safety_box(obs, actions)
            next_obs, rew, done, info = env.step(action=actions)
            if "intervene_action" in info:
                actions = info["intervene_action"]
            
            img = np.rot90(next_obs['agentview_image'], 2)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

            frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
            cv2.imshow("Agent View", frame_resized)
            img = np.rot90(next_obs['robot0_eye_in_hand_image'], 2)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

            frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
            cv2.imshow("Robot View", frame_resized)
            cv2.waitKey(1)

            # if check_impossible(env):
            #     next_obs, _ = env.reset()
            #     step = 0
            #     done = False
            #     rew = -1000

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

            if is_exit:
                cv2.destroyAllWindows()
                is_done = False
                is_save = False
                is_exit = False
                transitions = []
                print(env.env.env.sim.get_state().flatten())
                obs, _ = env.reset()
                break

            if is_done:
                cv2.destroyAllWindows()
                is_done = False
                is_save = False
                demos_count += 1
                pbar.update(1)
                if task_forward:
                    if positive:
                        forward_pos_trajectories += transitions
                    else:
                        forward_neg_trajectories += transitions

                transitions = []
                obs, _ = env.reset()
                break
    if len(forward_pos_trajectories) != 0:
        with open(fw_pos_file_path, "wb") as f:
            pkl.dump(forward_pos_trajectories, f)
            print(f"saved {len(forward_pos_trajectories)} transitions to {fw_pos_file_path}")

    if len(forward_neg_trajectories) != 0:
        with open(fw_neg_file_path, "wb") as f:
            pkl.dump(forward_neg_trajectories, f)
            print(f"saved {len(forward_neg_trajectories)} transitions to {fw_neg_file_path}")


    listener_1.stop()
    listener_2.stop()
    env.close()
    pbar.close()
