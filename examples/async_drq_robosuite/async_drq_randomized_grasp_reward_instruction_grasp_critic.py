#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from copy import deepcopy
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple
from typing import Union, List

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from transformers import AutoTokenizer, FlaxDistilBertModel
import torch
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

# from serl_launcher.agents.continuous.drq import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
from serl_launcher.utils.train_utils import concat_batches
from transformers import DistilBertTokenizer

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from serl_launcher.utils.launcher import (
    make_drq_agent_instruction,
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_sac_pixel_agent_hybrid_single_arm_instruction
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsRobosuiteWrapper, SERLObsRobosuiteInstructionWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierWrapper, GraspClassifierRobosuiteWrapper, GripperPenaltyWrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO

# from franka_env.envs.relative_env import RelativeFrame
# from franka_env.envs.wrappers import (
#     SpacemouseIntervention,
#     Quat2EulerWrapper,
#     FWBWFrontCameraBinaryRewardClassifierWrapper,
# )
import copy
import sys
import os
import cv2
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, OnScreenRenderEnv
from robosuite.wrappers import VisualizationWrapper
import keyboard  # 키 입력 감지 라이브러리
import threading
import time
import select
# import franka_env
pause = False
restart = False
stop_time = 0
FLAGS = flags.FLAGS

def key_listener():
    """
    키 입력을 감지하는 함수 (ESC를 누르면 정지, S를 누르면 재개)
    """
    global pause
    global restart
    global stop_time
    while True:
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:  # 입력 감지
            key = sys.stdin.read(1).lower()  # 1글자 입력 읽기
            if key == "q":  # ESC 키 대신 'q'를 사용 (터미널에서 ESC는 특수문자로 인식될 수 있음)
                pause = True
                restart = False
                stop_time = time.time()
                print("\n▶ 학습 정지됨. 's' 키를 눌러 다시 시작하세요.")
            elif key == "s":
                pause = False
                restart = True
                print("\n▶ 학습 재개됨.")

flags.DEFINE_string("env", "FrankaRobotiq-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 220, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 200000, "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_boolean("sparse_reward", False, "Sparse reward.")

flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
# "small" is a 4 layer convnet, "resnet" and "mobilenet" are frozen with pretrained weights
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_integer(
    "eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step"
)
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")
flags.DEFINE_string("fwbw", "fw", "forward or backward task")

# Checkpoints paths
flags.DEFINE_string("fw_ckpt_path", None, "Path to the fw checkpoint.")
flags.DEFINE_string("bw_ckpt_path", None, "Path to the bw checkpoint.")

# this is only used in actor node
flags.DEFINE_string(
    "fw_reward_classifier_ckpt_path",
    None,
    "Path to the fw reward classifier checkpoint.",
)
flags.DEFINE_string(
    "bw_reward_classifier_ckpt_path",
    None,
    "Path to the bw reward classifier checkpoint.",
)
flags.DEFINE_string(
    "grasp_reward_classifier_ckpt_path",
    None,
    "Path to the bw reward classifier checkpoint.",
)

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

id_to_task = {0: "fw"}
TrainerPortMapping = {
    "fw": {"port_number": 6700, "broadcast_port": 6701},
}


language_id_to_task = {0: "Pick up the bread from the light wood bin and place it in the dark wood bin.", 1: "Pick up the bread from the dark wood bin and place it in the light wood bin."}
language_id_to_text_feature = {0: 0, 1: 1}


xyz_bounding_box = np.array([[0.005, -0.14, 0.83], [0.19, 0.14, 0.92]])
dummy_action = np.array([0.] * 7)
fw_init_joint_state = np.array([0.10837591,  0.82401975, -0.08485613, -2.0482576,  -0.0779609,   3.14309925, 0.95097724])

bw_init_joint_state = np.array([ 0.29361669,  0.84783472,  0.08041102, -1.84587597, -0.29712924,  2.78031408, 1.38397813])


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def clip_safety_box(obs, action):
    """Clip the pose to be within the safety box while allowing movement back inside."""
    
    # 현재 엔드 이펙터 위치
    current_pos = obs['state']['tcp_pose'][:3]

    # 액션을 적용한 후 예상되는 위치
    next_pos = current_pos + (action[:3] / 10)

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
        if (
            obj_pos[2] < 0.8):
            print(f"불가능한 상태 감지: {obj_pos}")  # 디버깅 출력
            return True
    return False



# class FlaxDistilBERTInstructionEncoder(nn.Module):
#     pretrained_name: str = "distilbert-base-uncased"
#     output_dim: int = 256
#     max_length: int = 32

#     def setup(self):
#         # 토크나이저는 파라미터 트리에 포함되지 않도록 합니다.
#         self.tokenizer = DistilBertTokenizer.from_pretrained(self.pretrained_name)
#         # Flax DistilBERT 모델: 인스턴스 생성 후 params 속성으로 파라미터 접근 가능
#         self.encoder = FlaxDistilBertModel.from_pretrained(self.pretrained_name)
#         self.projection = nn.Dense(self.output_dim)

#     def __call__(self, instructions: Union[str, List[str]], train: bool = False):
#         if isinstance(instructions, str):
#             instructions = [instructions]
#         tokenized = self.tokenizer(
#             instructions,
#             return_tensors="jax",
#             padding="max_length",
#             max_length=self.max_length,
#             truncation=True
#         )
#         # train 모드일 때만 dropout_rng 생성. inference에서는 None으로 전달.
#         dropout_rng = self.make_rng('dropout') if train else None
#         # deterministic 인자 대신 train 인자를 사용합니다.
#         outputs = self.encoder(
#             **tokenized,
#             params=self.encoder.params,
#             dropout_rng=dropout_rng,
#             train=train
#         )
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
#         # projected = self.projection(cls_embedding)
#         return cls_embedding




##############################################################################


def actor(
    agents: OrderedDict[str, SACAgentHybridSingleArm],
    data_stores: OrderedDict[str, MemoryEfficientReplayBufferDataStore],
    env,
    sampling_rng,
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
        )
        for task in agents.keys():
            ckpt = checkpoints.restore_checkpoint(
                FLAGS.fw_ckpt_path if task == "fw" else FLAGS.bw_ckpt_path,
                agents[task].state,
                step=FLAGS.eval_checkpoint_step,
            )
            agents[task] = agents[task].replace(state=ckpt)

        success_count = {"fw": 0, "bw": 0}
        overall_success_count = 0
        cycle_time = {"fw": [], "bw": []}
        env.reset()
        #Initialize the state
        for i in range(10):
            env.step(dummy_action)
        reset_dict = {}
        reset_dict['init_state'] = init_state

        time_len = 1
        qpos_len = len(env.env.env.env.env.envsim.get_state().qpos)
        robot_qpos_init_state = env.env.env.env.env.env.sim.get_state().flatten()[time_len:time_len+9]
        robot_qvel_init_state = env.env.env.env.env.env.sim.get_state().flatten()[time_len+qpos_len:time_len+qpos_len+9]
        reset_dict['robot_qpos_init_state'] = robot_qpos_init_state
        reset_dict['robot_qvel_init_state'] = robot_qvel_init_state

        for _ in range(FLAGS.eval_n_trajs):
            for task_id, task_name in id_to_task.items():
                env.set_task_id(task_id)
                done = False
                current_step = 0
                start_time = time.time()
                while not done:
                    
                    actions = agents[task_name].sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    next_obs, reward, done, info = env.step(actions)
                    obs = next_obs
                    current_step += 1
                    if current_step == FLAGS.max_traj_length:
                        truncated = True
                        break
                    else:
                        truncated = False
                if reward:
                    dt = time.time() - start_time
                    cycle_time[task_name].append(dt)
                    print(f"{task_name}_cycle time: {dt} secs")
                last_state = env.env.env.env.env.env.env.env.sim.get_state().flatten()
                
                obs, _ = env.reset()
                time_len = 1
                qpos_len = len(env.env.env.env.env.env.env.env.sim.get_state().qpos)
                last_state[time_len:time_len+9] = reset_dict['robot_qpos_init_state']
                last_state[time_len+qpos_len:time_len+qpos_len+9] = reset_dict['robot_qvel_init_state']
                env.env.env.env.env.env.env.env.sim.set_state_from_flattened(last_state)
                obs = env.env.env.env.env.env.observation_()
                success_count[task_name] += reward
                print(reward)
                print(
                    f"{task_name}_success count: {success_count[task_name]} out of {FLAGS.eval_n_trajs}"
                )

            overall_success_count += reward
            print(
                f"overall_success count: {overall_success_count} out of {FLAGS.eval_n_trajs}"
            )
            print(f"average fw_cycle time: {np.mean(cycle_time['fw'])} secs")
            print(f"average bw_cycle time: {np.mean(cycle_time['bw'])} secs")

        return  # after done eval, return and exit
    
    global pause
    global restart
    global stop_time

    threading.Thread(target=key_listener, daemon=True).start()
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    clients = {
        task: TrainerClient(
            "actor_env",
            FLAGS.ip,
            make_trainer_config(**config),
            data_stores[task],
            wait_for_server=True,
        )
        for task, config in TrainerPortMapping.items()
    }

    # Function to update the fw agent with new params
    def update_params_fw(params):
        nonlocal agents
        agents["fw"] = agents["fw"].replace(
            state=agents["fw"].state.replace(params=params)
        )

    # Function to update the bw agent with new params
    def update_params_bw(params):
        nonlocal agents
        agents["bw"] = agents["bw"].replace(
            state=agents["bw"].state.replace(params=params)
        )

    clients["fw"].recv_network_callback(update_params_fw)

    env.set_task_id(0)
    obs, _ = env.reset()
    obs['instruction'] = language_id_to_text_feature[env.task_id]
    next_obs, reward, done, info = env.step(dummy_action)
    next_obs['instruction'] = language_id_to_text_feature[env.task_id]
    done = False
    init_state = env.env.env.env.env.env.env.sim.get_state().flatten()
    reset_dict = {}
    reset_dict['init_state'] = init_state
    reset_dict['fw_init_state'] = [2.53500000e+01, 7.95239493e-02, 8.06949949e-01, -4.88257703e-02,
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
    
    
    reset_dict['bw_init_state'] = [0, 2.33381969e-01, 7.74798222e-01, 4.43388729e-02,
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



    time_len = 1
    qpos_len = len(env.env.env.env.env.env.sim.get_state().qpos)
    robot_qpos_init_state = env.env.env.env.env.env.sim.get_state().flatten()[time_len:time_len+9]
    robot_qvel_init_state = env.env.env.env.env.env.sim.get_state().flatten()[time_len+qpos_len:time_len+qpos_len+9]
    reset_dict['robot_qpos_init_state'] = robot_qpos_init_state
    reset_dict['robot_qvel_init_state'] = robot_qvel_init_state
    reset_dict['bw_robot_qpos_init_state'] = [-1.70602805e-02, 2.04178504e-01,  3.70880593e-03, -2.58632465e+00, -7.92837093e-03,  2.94442322e+00,  8.04796788e-01,  2.08330000e-02, -2.08330000e-02]
    reset_dict['bw_robot_qvel_init_state'] = [1.00000000e+01, 1.00000000e+01, 1.00000000e+01,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 9.22317446e-02, -2.42341583e-01]

    current_step = 0
    # training loop
    timer = Timer()
    running_return = 0.0
    start_time = time.time()

    step = {"fw": 0}

    pbars = {
        v: tqdm.tqdm(
            total=FLAGS.max_steps,
            initial=0,
            desc=f"Training {v} actor",
            leave=True,
            dynamic_ncols=True,
        )
        for k, v in id_to_task.items()
    }

    while step["fw"] < FLAGS.max_steps:
        while pause:
            time.sleep(0.1)
        if restart:
            start_time = time.time() - stop_time + start_time
            restart = False
        timer.tick("total")
        current_step += 1
        task_name = "fw"
        with timer.context("sample_actions"):
            if step[task_name] < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agents[task_name].sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

                if "intervene_action" in info:
                    actions = info.pop("intervene_action")

        # Step environment
        with timer.context("step_env"):
            # actions = clip_safety_box(obs, actions)
            next_obs, reward, done, info = env.step(actions)
            next_obs['instruction'] = language_id_to_text_feature[env.task_id]
            frame1 = np.rot90(next_obs['agentview_image'], k=2)
            frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)  # BGR 변환
            frame_name = "BW" if env.task_id else "FW"
            cv2.imshow(frame_name+" Instruct Agent View", frame1_bgr)  # 첫 번째 창에 출력
            cv2.moveWindow(frame_name+" Instruct Agent View", 500, 300)  # "Robot View" 창을 화면의 (100, 100) 위치로 이동

            frame2 = np.rot90(next_obs['robot0_eye_in_hand_image'], k=2)
            frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)  # BGR 변환
            cv2.imshow(frame_name+" Instruct Eye in Hand View", frame2_bgr)  # 두 번째 창에 출력
            cv2.moveWindow(frame_name+" Instruct Eye in Hand View", 1000, 300)  # "Robot View" 창을 화면의 (100, 100) 위치로 이동

            frame3 = np.rot90(next_obs['robot0_eye_in_hand_left_image'], k=2)
            frame3_bgr = cv2.cvtColor(frame3, cv2.COLOR_RGB2BGR)  # BGR 변환
            cv2.imshow(frame_name+" Instruct Eye in Left Hand View", frame3_bgr)  # 두 번째 창에 출력
            cv2.moveWindow(frame_name+" Instruct Eye in Left Hand View", 0, 300)  # "Robot View" 창을 화면의 (100, 100) 위치로 이동

            cv2.waitKey(1)  # 1ms 대기

            step[task_name] += 1
            pbars[task_name].update(1)

            # override the action with the intervention action
            if check_impossible(env):
                done = False
                reward = np.asarray(reward, dtype=np.float32)
                # info = np.asarray(info)
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                running_return += reward
                if env.task_id:
                    next_obs, _ = env.reset()
                    # bw_init_state = reset_dict['bw_init_state']
                    # time_len = 1
                    # bw_init_state[time_len:time_len+9] = reset_dict['bw_robot_qpos_init_state']
                    # bw_init_state[time_len+qpos_len:time_len+qpos_len+9] = reset_dict['bw_robot_qvel_init_state']
                    env.env.env.env.env.env.sim.set_state_from_flattened(reset_dict['bw_init_state'])
                    env.env.env.env.env.env.robots[0].set_robot_joint_positions(bw_init_joint_state) 
                    next_obs = env.env.env.env.env.observation_()
                    # for i in range(15):
                    #     obs, rew1, done1, info1 = env.step(np.array([0.0, 1.0, 0, 0, 0, 0, 0]))
                    # for i in range(5):
                    #     obs, rew1, done1, info1 = env.step(np.array([0.0, 0.0, 0, 0, 0, 0.5, 0]))
                    # next_obs, reward, done, info = env.step(dummy_action)
                else:
                    next_obs, _ = env.reset()
                    env.env.env.env.env.env.sim.set_state_from_flattened(reset_dict['fw_init_state'])
                    env.env.env.env.env.env.robots[0].set_robot_joint_positions(fw_init_joint_state)
                    next_obs = env.env.env.env.env.observation_()
                
            else:
                reward = np.asarray(reward, dtype=np.float32)
                # info = np.asarray(info)
                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )

            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_stores[task_name].insert(transition)
            if current_step == FLAGS.max_traj_length-1:
                truncated = True
            else:
                truncated = False
            obs = next_obs
            if done or truncated:
                next_task_id = env.task_id
                print("Final state Reward: ", reward)
                if done:
                    print("Robosuite task success!")
                    next_task_id = (env.task_id + 1) % 2
                if env.task_id == next_task_id:
                    done_check = False
                    fw_done = 0
                    bw_done = 0
                else:
                    done_check = True
                    if env.task_id:
                        fw_done = 0
                        bw_done = 1
                    else:
                        fw_done = 1
                        bw_done = 0

                print(f"transition from {env.task_id} to next task: {next_task_id}")
                if env.task_id:
                    stats['fw_reward'] = -1
                    stats['bw_reward'] = running_return
                else:
                    stats['fw_reward'] = running_return
                    stats['bw_reward'] = -1
                elapsed_time = time.time() - start_time
                wandb_logger.log({
                    "actor/running_return_fw": stats['fw_reward'],
                    "actor/running_return_bw": stats['bw_reward'],
                    "actor/running_return_avg_fw": stats['fw_reward'] / current_step,
                    "actor/running_return__avg_bw": stats['bw_reward'] / current_step,
                    "actor/step": current_step,
                    "actor/task": env.task_id,
                    "actor/fw_done": fw_done,
                    "actor/bw_done": bw_done,
                    "actor/elapsed_time_sec": elapsed_time,  # 초 단위
                    "actor/elapsed_time_min": elapsed_time / 60,  # 분 단위
                    "actor/elapsed_time_hr": elapsed_time / 3600,  # 시간 단위
                }, step=(step["fw"]))
                current_step = 0

                env.set_task_id(next_task_id)

                stats = {f"{task_name}_train": info}  # send stats to the learner to log
                stats["env_steps"] = step[task_name]
                clients[task_name].request("send-stats", stats)
                running_return = 0.0
                last_state = env.env.env.env.env.env.env.sim.get_state().flatten()
                
                obs, _ = env.reset()
                
                # last_state에서 로봇 팔 위치만 재위치
                if next_task_id:
                    # time_len = 1
                    # qpos_len = len(env.env.env.env.env.env.sim.get_state().qpos)
                    # last_state[time_len:time_len+9] = reset_dict['bw_robot_qpos_init_state'] 
                    # last_state[time_len+qpos_len:time_len+qpos_len+9] = reset_dict['bw_robot_qvel_init_state']'
                    # env.env.env.env.env.env.sim.set_state_from_flattened(last_state)
                    env.env.env.env.env.env.sim.set_state_from_flattened(reset_dict['bw_init_state'])
                    env.env.env.env.env.env.robots[0].set_robot_joint_positions(bw_init_joint_state)

                    obs = env.env.env.env.env.observation_()
                    
                    
                else:
                    # time_len = 1
                    # qpos_len = len(env.env.env.env.env.env.sim.get_state().qpos)
                    # last_state[time_len:time_len+9] = reset_dict['robot_qpos_init_state'] + np.random.uniform(-0.05, 0.05, 9)
                    # last_state[time_len+qpos_len:time_len+qpos_len+9] = reset_dict['robot_qvel_init_state'] + np.random.uniform(-0.05, 0.05, 9)
                    # env.env.env.env.env.env.sim.set_state_from_flattened(last_state)
                    env.env.env.env.env.env.sim.set_state_from_flattened(reset_dict['fw_init_state'])
                    env.env.env.env.env.env.robots[0].set_robot_joint_positions(fw_init_joint_state)

                    obs = env.env.env.env.env.observation_()

        timer.tock("total")
        for name, task_step in step.items():
            if task_step % FLAGS.steps_per_update == 0:
                clients[name].update()
            if task_step % FLAGS.log_period == 0:
                stats = {f"{name}_timer": timer.get_average_times()}
                clients[name].request("send-stats", stats)

    for pbar in pbars.values():
        pbar.close()


##############################################################################


def learner(rng, agent: SACAgentHybridSingleArm, replay_buffer, demo_buffer):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    global pause
    global restart
    global stop_time
    threading.Thread(target=key_listener, daemon=True).start()
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0
    env_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        nonlocal env_steps
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            if "env_steps" in payload:
                env_steps = payload["env_steps"]
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(
        make_trainer_config(**TrainerPortMapping[FLAGS.fwbw]),
        request_callback=stats_callback,
    )
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc=f"Filling up {FLAGS.fwbw} replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
    train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    pbar = tqdm.tqdm(
        total=FLAGS.max_steps,
        initial=0,
        desc=f"Updating {FLAGS.fwbw} learner",
        leave=True,
    )
    while update_steps < FLAGS.max_steps:
        while pause:
            time.sleep(0.1)

        if not update_steps < env_steps:
            time.sleep(1)
            continue
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.critic_actor_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
            )

        # publish the updated network
        if update_steps > 0 and update_steps % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=update_steps,
                keep=100,
                overwrite=True,
            )

        update_steps += 1
        pbar.update(1)


##############################################################################


def main(_):
    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

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
    task_description = options['env_name'] + "_" + options['robots']
    # initialize the task
    
    controller_name = choose_controller()

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    options['controller_configs']['position_limits'] = xyz_bounding_box
    orientation_limit = np.array([
        [-np.deg2rad(180), -np.deg2rad(180), -np.deg2rad(180)],
        [ np.deg2rad(180),  np.deg2rad(180),  np.deg2rad(180)]
    ])
    options['controller_configs']['orientation_limits'] = orientation_limit
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

    env = SERLObsRobosuiteInstructionWrapper(env)
    env = SpacemouseInterventionLIBERO(env)

    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)

    env = FrontCameraLIBEROWrapper(env)
    image_keys = [key for key in env.observation_space.keys() if key != "state" and key != "instruction"]
    language_keys = ["instruction"]
    env_image_observation_space = env.image_observation_space
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = FlaxDistilBertModel.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer([language_id_to_task[0], language_id_to_task[1]], return_tensors="jax")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    feature = last_hidden_states[:, 0, :]
    language_id_to_text_feature[0] = feature[0]
    language_id_to_text_feature[1] = feature[1]
    env = GripperPenaltyWrapper(env, penalty=-0.05)

    if FLAGS.actor:
        front_image_keys = [
            k for k in env.front_observation_space.keys() if "state" not in k
        ]

        from serl_launcher.networks.reward_classifier import load_classifier_func

        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)

        if (
            not FLAGS.fw_reward_classifier_ckpt_path
            or not FLAGS.bw_reward_classifier_ckpt_path
        ):
            raise ValueError(
                "Must provide both fw and bw reward classifier ckpt paths for actor"
            )

        fw_classifier_func = load_classifier_func(
            key=key,
            sample=env.front_observation_space.sample(),
            image_keys=front_image_keys,
            checkpoint_path=FLAGS.fw_reward_classifier_ckpt_path,
        )
        rng, key = jax.random.split(rng)
        bw_classifier_func = load_classifier_func(
            key=key,
            sample=env.front_observation_space.sample(),
            image_keys=front_image_keys,
            checkpoint_path=FLAGS.bw_reward_classifier_ckpt_path,
        )
        if FLAGS.sparse_reward:
            env = FWBWFrontCameraBinaryRewardClassifierWrapper(
                env, fw_classifier_func, bw_classifier_func
            )
        else:
            env = FWBWFrontCameraRewardClassifierWrapper(
                env, fw_classifier_func, bw_classifier_func
            )
        # env = RecordEpisodeStatistics(env)
        # rng, key = jax.random.split(rng)
        # grasp_classifier_func = load_classifier_func(
        #         key=key,
        #         sample=env_image_observation_space.sample(),
        #         image_keys=image_keys,
        #         checkpoint_path=FLAGS.grasp_reward_classifier_ckpt_path
        #     )
        # env = GraspClassifierRobosuiteWrapper(env, grasp_classifier_func)

        agents = OrderedDict()
        for k, v in id_to_task.items():
            rng, sampling_rng = jax.random.split(rng)
            agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm_instruction(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=image_keys,
                language_keys=language_keys,
                encoder_type=FLAGS.encoder_type,
            )
            # replicate agent across devices
            # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
            agent: SACAgentHybridSingleArm = jax.device_put(
                jax.tree_map(jnp.array, agent), sharding.replicate()
            )
            agents[v] = agent

    else:
        rng, sampling_rng = jax.random.split(rng)
        agent: SACAgentHybridSingleArm = make_drq_agent_instruction(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=image_keys,
            language_keys=language_keys,
            encoder_type=FLAGS.encoder_type,
        )
        # replicate agent across devices
        # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
        agent: SACAgentHybridSingleArm = jax.device_put(
            jax.tree_map(jnp.array, agent), sharding.replicate()
        )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=40000,
            image_keys=image_keys,
        )
        import pickle as pkl

        demo_path = FLAGS.demo_path  # 폴더 경로
        if os.path.isdir(demo_path):  # 경로가 폴더인지 확인
            demo_files = sorted([os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith(".pkl")])
        else:
            demo_files = [demo_path]  # 단일 파일인 경우 리스트로 처리
        for demo_file in demo_files:
            if "fw" in demo_file:
                instruction = language_id_to_text_feature[0]
            else:
                instruction = language_id_to_text_feature[1]
            with open(demo_file, "rb") as f:
                trajs = pkl.load(f)
                for traj in trajs:
                    for tra in traj:
                        tra['observations']['state'] = {k: v for k, v in tra['observations']['state'].items() if k in ['tcp_pose', 'robot0_gripper_qpos', 'robot0_gripper_qvel']}
                        tra['next_observations']['state'] = {k: v for k, v in tra['next_observations']['state'].items() if k in ['tcp_pose', 'robot0_gripper_qpos', 'robot0_gripper_qvel']}
                        tra['observations']['instruction'] = instruction
                        tra['next_observations']['instruction'] = instruction
                        traj['actions'][3:6] = np.clip(traj['actions'][3:6], 0.0, 0.0)
                        # if keys in tra is in [bread_pos, next_bread_pos], delete these keys
                        if 'bread_pos' in traj:
                            del traj['bread_pos']
                        if 'next_bread_pos' in traj:
                            del traj['next_bread_pos']
                        demo_buffer.insert(tra)
            print(f"Loaded {demo_file}, buffer size: {len(demo_buffer)}")

        print(f"Total demo buffer size: {len(demo_buffer)}")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_stores = OrderedDict(
            {name: QueuedDataStore(2000) for name in id_to_task.values()}
        )
        # actor loop
        print_green("starting actor loop")
        actor(agents, data_stores, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
