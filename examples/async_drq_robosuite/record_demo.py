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

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsVisualizationLIBEROWrapper, RelativeFrame, RelativeRobosuiteFrame, Quat2EulerWrapper, SafeControlWrapper, SERLObsRobosuiteWrapper, SERLObsRobosuiteInstructionWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, GripperPenaltyWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
import argparse
from libero.libero import benchmark
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
import cv2

xyz_bounding_box = np.array([[0.045, -0.135, 0.84], [0.15, 0.14, 0.88]]) #0.21이 넘으면 넘기기
dummy_action = np.array([0.] * 7)
fw_init_joint_state = np.array([0.01318197, 0.70407622, 0.01265135, -2.1516366, -0.01592278, 3.02288862, 0.84461911])
fw_init_qpos = np.array([-0.11104011, 0.83457593, 0.11598836, -1.87966513, 0.03060678, 2.64441078, -0.84230056])

# bw_init_joint_state = np.array([ 0.29361669,  0.84783472,  0.08041102, -1.84587597, -0.29712924,  2.78031408, 1.38397813])
# bw_init_joint_state = np.array([ 0.24903555,  0.78565825,  0.05100037, -1.97534394, -0.26406667,  2.87867408, 1.27396497])
# bw_init_joint_state = np.array([ 0.2775879,   0.83051554,  0.03900583, -1.93557079, -0.25035431,  2.90343858, 1.24637345])
bw_init_joint_state = np.array([ 0.2920244,   0.7800007,   0.01457994, -1.9885669,  -0.29130036,  2.93142096, 1.3405883 ])
# bw_init_qpos = np.array([0.04973374, 0.86776707, 0.26348287, -1.77921961 , -0.16386197, 2.58488172, -0.41709979])
# bw_init_qpos = np.array([0.33906145, 0.84236467, -0.01812242, -1.817323, -0.21512965, 2.63209568, 2.89666725])
# bw_init_qpos = np.array([0.03176846, 0.85611438, 0.24880941, -1.80200212, -0.14938099, 2.5967833, -0.44818929])
bw_init_qpos = np.array([0.03596594, 0.89493965, 0.2427182, -1.77591072, -0.1494936, 2.61021326, -0.46079463])

# [ 0.04973374  0.86776707  0.26348287 -1.77921961 -0.16386197  2.58488172
#  -0.41709979]
fw_init_robot_qpos = np.array([0.02032033,  0.67931426, -0.01355595, -2.17666628, -0.04708652,  2.94202914, 0.89655496,  0.03998476, -0.04000051])
fw_init_robot_qvel =np.array([-7.70646891e-05, -4.20885315e-06,  3.35458312e-05,  1.08548752e-05, -6.81516247e-05,  1.83453699e-05, -8.88580400e-06,  1.80590822e-06, 9.04253834e-10])

bw_init_robot_qpos = np.array([0.24976532,  0.82012957, 0.0565687, -1.91029751,  -0.28946547,   2.83323272, 1.37469773, 0.03998855, -0.0400003])
bw_init_robot_qvel = np.array([-9.74769020e-12,  7.78071992e-12,  2.56474873e-11,  1.21211398e-12, 3.00693306e-11,  3.05974069e-11, -3.10940976e-12,  3.24059354e-06,-3.40507135e-11])



fw_init_state = [0, 7.95239493e-02, 8.06949949e-01, -4.88257703e-02,
                 -2.08114917e+00, -6.27997959e-02, 3.07558028e+00, 8.56096827e-01,
                 3.99591728e-02, -4.00015793e-02, 1.00791971e+01, 1.13846351e+01,
                 6.09910397e-02, 9.99997059e-01, 4.35046661e-06, 7.90380443e-06,
                 2.42506644e-03, 9.06768269e-02, -7.99428439e-02, 8.42216565e-01,
                 -1.68134668e-02, 2.53077704e-04, -5.87659523e-04, 9.99858439e-01,
                 1.00245190e+01, 1.08035536e+01, 1.49751653e-02, 7.07073837e-01,
                 7.07076187e-01, 6.70544400e-03, 6.70009138e-03, 1.00002095e+01,
                 5.54282228e+00, 2.48506508e-02, 7.04007259e-01, 7.06576214e-01,
                 2.58957269e-02, -6.67326323e-02, -2.47226457e-04, -3.68408112e-06,
                 1.84050590e-05, 9.07866976e-06, -7.34896772e-05, 1.32321976e-05,
                 4.30203895e-05, -4.60330530e-08, 7.18376246e-09, -5.18947632e-04,
                 4.98846996e-04, -8.09547200e-06, -7.76573974e-03, -7.99488241e-03,
                 -4.18165794e-08, -1.65012296e-02, -6.69404019e-03, 5.38753888e-04,
                 -2.60209389e-01, 3.71046125e-01, 5.72020021e-04, 9.22251837e-05,
                 1.70278050e-05, 8.33745167e-08, -1.00343201e-03, -2.68290253e-08,
                 -6.06977047e-03, -7.97456131e-04, 9.32601882e-05, 1.56990276e-04,
                 -1.64960838e-03, -2.14236468e-04, 2.32712723e-02]

# bw_init_state = [0, 2.33381969e-01, 7.74798222e-01, 4.43388729e-02,
#                  -1.96652932e+00, -3.06101842e-01, 2.87887416e+00, 1.32885817e+00,
#                  1.99904830e-02, -2.00095172e-02, 1.00791950e+01, 1.13846371e+01,
#                  6.09910177e-02, 9.99997060e-01, -1.21684063e-05, -9.32503912e-06,
#                  2.42493851e-03, 9.74456569e-02, 8.50200195e-02, 8.42217757e-01,
#                  -7.08860685e-01, 6.32028161e-05, 8.54208729e-04, 7.05347996e-01,
#                  1.00245188e+01, 1.08035536e+01, 1.49751660e-02, 7.07075981e-01,
#                  7.07074058e-01, 6.69933054e-03, 6.70468159e-03, 1.00001747e+01,
#                  5.54282591e+00, 2.48506576e-02, 7.04012430e-01, 7.06576861e-01,
#                  2.58456167e-02, -6.66906417e-02, -8.31817350e-06, 3.25078220e-03,
#                  -1.41675255e-05, 3.35534449e-03, 5.63339008e-06, -5.76267072e-05,
#                  -5.01127471e-05, 1.70930923e-07, 1.71123933e-07, -7.17375305e-04,
#                  6.89141726e-04, -2.30153536e-05, -1.09486202e-02, -1.12836598e-02,
#                  -6.13861716e-08, 7.08133305e-03, 5.80271131e-03, 1.34840751e-03,
#                  -7.71236484e-02, -2.43378966e-01, 4.45034949e-04, -9.23008839e-05,
#                  -1.70283429e-05, -1.43013288e-07, 1.00338969e-03, -4.86727263e-08,
#                  6.07472449e-03, -2.44123488e-04, 3.12173503e-05, 6.98448274e-05,
#                  -5.46653607e-04, -7.00593440e-05, 5.74352264e-03]

bw_init_state = [0, 2.33381969e-01, 7.74798222e-01, 4.43388729e-02,
                 -1.96652932e+00, -3.06101842e-01, 2.87887416e+00, 1.32885817e+00,
                 1.99904830e-02, -2.00095172e-02, 1.00791950e+01, 1.13846371e+01,
                 6.09910177e-02, 9.99997060e-01, -1.21684063e-05, -9.32503912e-06,
                 2.42493851e-03, 9.74456569e-02, 8.50200195e-02, 8.42217757e-01,
                 -7.08860685e-01, 6.32028161e-05, 8.54208729e-04, 7.05347996e-01,
                 1.00245188e+01, 1.08035536e+01, 1.49751660e-02, 7.07075981e-01,
                 7.07074058e-01, 6.69933054e-03, 6.70468159e-03, 1.00001747e+01,
                 5.54282591e+00, 2.48506576e-02, 7.04012430e-01, 7.06576861e-01,
                 2.58456167e-02, -6.66906417e-02, 0, 0,
                    0, 0, 0, 0,
                    0, 0 ,0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0, 
                    0, 0, 0, 0, 
                    0, 0, 0, 0,
                    0 ,0, 0, 0,
                    0 ,0, 0,
                 ]
def clip_safety_box(obs, action):
    """Clip the pose to be within the safety box while allowing movement back inside."""
    
    # 현재 엔드 이펙터 위치
    current_pos = obs['state']['tcp_pose'][:3]

    # 액션을 적용한 후 예상되는 위치
    next_pos = current_pos + (action[:3] / 100)

    # Safety Box 범위
    min_bounds, max_bounds = xyz_bounding_box  # (array([x_min, y_min, z_min]), array([x_max, y_max, z_max]))

    # Safety Box를 벗어나려는 방향 확인
    out_of_bounds_low = next_pos < min_bounds
    out_of_bounds_high = next_pos > max_bounds

    # **들어오는 액션인지 확인**
    moving_inward = (action[:3] > 0) & out_of_bounds_low | (action[:3] < 0) & out_of_bounds_high
    clipped_action = copy.deepcopy(dummy_action)
    clipped_action[3:] = action[3:]
    # Safety Box를 벗어나려 하고, 안으로 들어오는 방향이 아니라면 -> 이동 제한
    clipped_action[:3] = np.where(out_of_bounds_low & ~moving_inward, 0, action[:3])
    clipped_action[:3] = np.where(out_of_bounds_high & ~moving_inward, 0, clipped_action[:3])

    return clipped_action


def check_impossible(env):
    obs = env.get_state_obs()
    
    # 객체들의 위치 정보를 리스트로 저장
    obj_pos_list = [
        obs['Bread_pos'],
    ]
    
    # 불가능한 상황 판단
    for obj_pos in obj_pos_list:
        # xyz_bounding_box 밖이면 불가능한 상태
        if obj_pos[2] < 0.8 or (obj_pos[0] < -0.003 or obj_pos[0] > 0.205):
            print(f"불가능한 상태 감지: {obj_pos}")  # 디버깅 출력
            return True
    return False

if __name__ == "__main__":
    
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
    options['controller_configs']['position_limits'] = xyz_bounding_box

    orientation_limit = np.array([
        [np.deg2rad(135), -np.deg2rad(60), -np.deg2rad(0)],
        [ np.deg2rad(180),  np.deg2rad(60),  np.deg2rad(180)]
    ])
    # orientation_limit = np.array([
    #     [np.deg2rad(135), -np.deg2rad(180), -np.deg2rad(0)],
    #     [ np.deg2rad(180),  np.deg2rad(180),  np.deg2rad(180)]
    # ])
    # orientation_limits = [
    #     [-3.14, -1.7, -2.5],  # 각 차원의 최소값 (rad)
    #     [ 3.14,  1.7,  2.5]   # 각 차원의 최대값 (rad)
    # ]
    # options['controller_configs']['orientation_limits'] = orientation_limit

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
    env = SpacemouseInterventionLIBERO(env)
    env = RelativeRobosuiteFrame(env)
    env = Quat2EulerWrapper(env)
    env = SafeControlWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraLIBEROWrapper(env)
    env = GripperPenaltyWrapper(env)
    obs, _ = env.reset(task_id=0)
    time_len = 1
    reset_dict = {}
    last_state = env.env.env.env.env.env.sim.get_state().flatten()
    qpos_len = len(env.env.env.env.env.sim.get_state().qpos)
    reset_dict['robot_qpos_init_state'] = last_state[time_len:time_len+9]
    reset_dict['robot_qvel_init_state'] = last_state[time_len+qpos_len:time_len+qpos_len+9]    
    reset_dict['bw_robot_qpos_init_state'] = [-1.70602805e-02, 2.04178504e-01,  3.70880593e-03, -2.58632465e+00, -7.92837093e-03,  2.94442322e+00,  8.04796788e-01,  2.08330000e-02, -2.08330000e-02]
    reset_dict['bw_robot_qvel_init_state'] = [1.00000000e+01, 1.00000000e+01, 1.00000000e+01,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.22317446e-02, -2.42341583e-01]

    scale_factor = 2
    
    # next_obs1, rew1, done1, info1 = env.step(action=np.zeros((7,)))
    demos_count = 0
    demos_needed = 40

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

    pbar = tqdm(total=demos_needed, desc="demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fw_pos_file_name = f"fw_pos_{task_description}_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    
    bw_pos_file_name = f"bw_pos_{task_description}_demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    
    fw_pos_file_name = f"fw_pos__demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")
    bw_pos_file_name = f"bw_pos__demos_{uuid}.pkl".replace(" ", "_").replace("-", "_")

    file_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "demos")
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    fw_pos_file_path = os.path.join(file_dir, 'fw/'+fw_pos_file_name)
    bw_pos_file_path = os.path.join(file_dir, 'bw/'+bw_pos_file_name)

    transitions = []
    actions = np.zeros((7,))
    # actions[0] = 0.1
    while demos_count < demos_needed:
        print("Current the number of task: forward {}, backward {}".format(
            len(forward_pos_trajectories), len(backward_pos_trajectories)
        ))
        task_id = input("Enter forward (z), backward (x), save and exit (q): ")

        print("Save buttom: F1, Done button: ESC, Exit button: F2")
        if task_id == 'z':
            obs, _ = env.reset(task_id=0)
            task_forward = True
            positive = True
            # initial_state = env.env.env.env.env.sim.get_state().flatten()
            time_len = 1
            qpos_len = len(env.env.env.env.sim.get_state().qpos)
            last_state[time_len:time_len+9] = fw_init_robot_qpos
            last_state[time_len+qpos_len:time_len+qpos_len+9] = fw_init_robot_qvel
            # env.env.env.env.env.env.sim.set_state_from_flattened(fw_init_state)
            env.env.env.env.env.env.sim.set_state_from_flattened(last_state)

            # env.env.env.env.env.env.robots[0].set_robot_joint_positions(fw_init_joint_state)
            env.env.env.env.env.env.robots[0].sim.forward()
            env.env.env.env.env.env.robots[0].reset(deterministic=True, init_qpos_=fw_init_qpos)
            for i in range(10):
                gripper_open_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                obs, _, _, _ = env.step(gripper_open_action)      
            last_state = env.env.env.env.env.env.sim.get_state().flatten()
            obs['state']['Bread_pos'] = env.get_state_obs()['Bread_pos']
            actions = dummy_action

        elif task_id == 'x':
            obs, _ = env.reset(task_id=1)

            task_forward = False
            positive = True
            # initial_state = env.env.env.env.env.sim.get_state().flatten()
            time_len = 1
            last_state[time_len:time_len+9] = bw_init_robot_qpos
            last_state[time_len+qpos_len:time_len+qpos_len+9] = bw_init_robot_qvel
            # env.env.env.env.env.env.sim.set_state_from_flattened(bw_init_state)
            env.env.env.env.env.env.sim.set_state_from_flattened(last_state)
            env.env.env.env.env.env.sim.forward()

            # env.env.env.env.env.env.robots[0].set_robot_joint_positions(bw_init_joint_state)
            # env.env.env.env.env.env.robots[0].sim.forward()
            env.env.env.env.env.env.robots[0].reset(deterministic=True, init_qpos_=bw_init_qpos)
            for i in range(10):
                gripper_open_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                obs, _, _, _ = env.step(gripper_open_action)      
            obs['state']['Bread_pos'] = env.get_state_obs()['Bread_pos']
            actions = dummy_action
            # for i in range(15):
            #     obs, rew1, done1, info1 = env.step(np.array([0.0, 1.0, 0, 0, 0, 0, 0]))
            
            # for i in range(50):
            #     obs, rew1, done1, info1 = env.step(dummy_action)
        
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
                last_state = env.env.env.env.env.env.sim.get_state().flatten()
                obs, _ = env.reset(task_id=0 if task_forward else 1)
                env.env.env.env.env.env.sim.set_state_from_flattened(last_state)
                obs = env.env.env.env.observation_()
                step = 0
                action = dummy_action
                obs['state']['Bread_pos'] = env.get_state_obs()['Bread_pos']
            # actions = clip_safety_box(obs, actions)
            # import pdb; pdb.set_trace()
            state_obs = env.get_state_obs()
            obs_bread_pos = state_obs['Bread_pos']
            obs_robot_pos = state_obs['robot0_eef_pos']
            # print(obs_robot_pos)
            # if step < 500:
                # actions = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            next_obs, rew, done, info = env.step(action=actions)
            # print(next_obs)
            # if 'intervene_action' in info:
            #     print(info['intervene_action'])
            # print(info)
            # print(actions)
            # print(actions)
            # print(next_obs['state']['tcp_pose'])
            # print(info)
            # print(env.env.env.env.env.env.env.env.robots[0].recent_qpos.current)
            next_state_obs = env.get_state_obs()
            next_obs_bread_pos = next_state_obs['Bread_pos']
            next_obs_robot_pos = next_state_obs['robot0_eef_pos']
            # print(next_obs_bread_pos)
            # print(next_obs['state']['robot0_gripper_qpos'])
            # print(info['grasp_penalty'])
            #최대로 닫혀있을 때 [ 0.00048117 -0.00051859]
            #최대로 열려있을 때 [ 0.03999851 -0.03997442]
            #if self.left:  # open gripper
                # gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            # gripper_action 이 - 라면 open gripper!
            #state['robot0_gripper_qpos]는 [양수, 음수]
            # print(next_obs['state']['robot0_gripper_qvel'])
            # import pdb; pdb.set_trace()
            # print("qpos:", env.env.env.env.env.env.sim.get_state().qpos[:9])
            # print("qvel:", env.env.env.env.env.env.sim.get_state().qvel[:9])
            # env.env.env.env.env.robots[0].set_robot_eef_positions(np.array([-0.04, -0.2,  1.0,  0.99737413, -0.01520812, 0.07078563, -0.00171779]))
            # print(env.get_state_obs())
            # list1 = []
            # for i in range(4):
            #     list1.append(np.rad2deg(next_obs['state']['tcp_pose'][i+3]))
            # print(list1)
            
            # from scipy.spatial.transform import Rotation as R

            # # 이미 [qx, qy, qz, qw] 순서일 경우
            # q = next_obs['state']['tcp_pose'][3:7]
            # r = R.from_quat(q)
            # euler_angles = r.as_euler('xyz', degrees=False)
            # print("Euler angles:", euler_angles)
            # print(next_obs['state']['robot0_gripper_qpos'])
            # print(next_obs['state']['robot0_gripper_qpos'][0] - next_obs['state']['robot0_gripper_qpos'][1])
            # print(next_obs['state']['robot0_gripper_qvel'])
            if "intervene_action" in info:
                actions = info["intervene_action"]
            # print(actions)
            # img = np.rot90(next_obs['robot0_eye_in_hand_left_image'], 3)
            frame_bgr = cv2.cvtColor(next_obs['robot0_eye_in_hand_left_image'], cv2.COLOR_RGB2BGR)  # BGR 변환

            frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
            cv2.imshow("Robot 2 View", frame_resized)
            cv2.moveWindow("Robot 2 View", 1000, 300)  # "Robot View" 창을 화면의 (100, 100) 위치로 이동

            # img = np.rot90(next_obs['robot0_eye_in_hand_image'], 1)
            frame_bgr = cv2.cvtColor(next_obs['robot0_eye_in_hand_image'], cv2.COLOR_RGB2BGR)  # BGR 변환
        
            frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
            cv2.imshow("Robot View", frame_resized)
            cv2.moveWindow("Robot View", 1800, 300)  # "Robot View" 창을 화면의 (100, 100) 위치로 이동
            # img = np.rot90(next_obs['agentview_image'], 2)
            frame_bgr = cv2.cvtColor(next_obs['agentview_image'], cv2.COLOR_RGB2BGR)  # BGR 변환

            frame_resized = cv2.resize(frame_bgr, (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor))
            cv2.imshow("Agent View", frame_resized)
            cv2.waitKey(1)

            if check_impossible(env):
                next_obs, _ = env.reset()
                step = 0
                done = False
                next_obs_bread_pos = env.get_state_obs()['Bread_pos']

            xyz_bounding_box = np.array([[0.005, -0.14, 0.83], [0.19, 0.14, 0.92]])
            
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
            

            if next_obs_bread_pos[0] > 0.005 and next_obs_bread_pos[0] < 0.19 and next_obs_bread_pos[2] > 0.81 and next_obs_bread_pos[2] < 0.92:
                if next_obs_bread_pos[1] > 0.03 and next_obs_bread_pos[1] < 0.14 and task_forward == True:
                    rew = 1.0
                    done = 1.0
                elif next_obs_bread_pos[1] < -0.025 and next_obs_bread_pos[1] > -0.14 and task_forward == False:
                    rew = 1.0
                    done = 1.0
                else:
                    if is_grasping_bread(next_obs_robot_pos, obs['state']['robot0_gripper_qpos'], next_obs_bread_pos):
                        grasp_success = 1
                        grasp_reward = 0.5
                        if task_forward:
                            penalty = -np.abs(next_obs_robot_pos[1] - 0.07)
                            penalty_bread = -np.abs(next_obs_bread_pos[1] - 0.07)
                            
                        else:
                            penalty = -np.abs(next_obs_robot_pos[1] - (-0.07))
                            penalty_bread = -np.abs(next_obs_bread_pos[1] - (-0.07))
                        penalty += -np.abs(next_obs_robot_pos[0] - 0.095)
                        # print(np.abs(np.clip(obs['state']['tcp_pose'][2], 0, 0.88) - 0.80)*2)
                        penalty += np.abs(np.clip(next_obs_robot_pos[2], 0, 0.88) - 0.80)*3
                        grasp_reward += penalty
                        grasp_reward += penalty_bread
                        rew = grasp_reward
                    else:
                        grasp_success = 0
                        rew = -np.linalg.norm(next_obs_bread_pos - next_obs_robot_pos) * 5.0
                    done = 0.0
            else:
                rew = 0.0
                done = 0.0
            # -np.linalg.norm(next_obs_bread_pos - next_obs['state']['tcp_pose'][:3]) * 10.0 > -0.35
            # print(is_grasping_bread(next_obs, next_obs_bread_pos))
            # if next_obs_bread_pos[1] > 0.03 and next_obs_bread_pos[1] < 0.14 and next_obs_bread_pos[2] < 0.88 and task_forward == True:
            #     rew = 1.0
            #     done = 1.0
            # elif next_obs_bread_pos[1] < -0.025 and next_obs_bread_pos[1] > -0.14 and next_obs_bread_pos[2] < 0.88 and task_forward == False:
            #     rew = 1.0
            #     done = 1.0
            # else:
            #     rew = 0.0
            #     done = 0.0
            # else:
            #     if next_obs_bread_pos[2] > 0.846:
            #         grasp_reward = 0.5
            #         if task_forward:
            #             penalty = -np.abs(next_obs['state']['tcp_pose'][1] - 0.07)
            #             penalty_bread = -np.abs(next_obs_bread_pos[1] - 0.07)
                        
            #         else:
            #             penalty = -np.abs(next_obs['state']['tcp_pose'][1] - (-0.07))
            #             penalty_bread = -np.abs(next_obs_bread_pos[1] - (-0.07))
            #         grasp_reward += penalty
            #         grasp_reward += penalty_bread
            #         rew = grasp_reward
            #     else:
            #         rew = -np.linalg.norm(next_obs_bread_pos - obs['state']['tcp_pose'][:3]) * 5.0
            #     done = 0.0
            
            # print(info['grasp_penalty'])
            # print(rew, done)
            # if done:
                # is_done = True
            if is_save:
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - done,
                        dones=done,
                        grasp_penalty=info['grasp_penalty'],
                        bread_pos=obs_bread_pos,
                        next_bread_pos=next_obs_bread_pos,
                        obs_robot_pos=obs_robot_pos,
                        next_obs_robot_pos=next_obs_robot_pos,
                        
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
                print(env.env.env.env.sim.get_state().flatten())
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
                    if positive:
                        backward_pos_trajectories += transitions
                transitions = []
                last_state = env.env.env.env.env.sim.get_state().flatten()
                break

    if len(forward_pos_trajectories) != 0:
        with open(fw_pos_file_path, "wb") as f:
            pkl.dump(forward_pos_trajectories, f)
            print(f"saved {len(forward_pos_trajectories)} transitions to {fw_pos_file_path}")

    
    if len(backward_pos_trajectories) != 0:
        with open(bw_pos_file_path, "wb") as f:
            pkl.dump(backward_pos_trajectories, f)
            print(f"saved {len(backward_pos_trajectories)} transitions to {bw_pos_file_path}")

    listener_1.stop()
    listener_2.stop()
    env.close()
    pbar.close()
