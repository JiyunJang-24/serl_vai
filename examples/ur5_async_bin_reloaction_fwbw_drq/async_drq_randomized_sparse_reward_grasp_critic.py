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

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
    make_sac_pixel_agent_hybrid_single_arm,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper, SERLObsRobosuiteWrapper, RelativeFrame, Quat2EulerWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper, GraspClassifierWrapper, GraspClassifierRobosuiteWrapper, GripperPenaltyWrapper, GripperPenaltyUR5Wrapper
from serl_launcher.wrappers.spacemouse import SpacemouseInterventionLIBERO, SpacemouseInterventionUR5
import ur_env
from ur_env.envs.wrappers import SpacemouseIntervention, ToMrpWrapper, ObservationRotationWrapper, Quat2EulerWrapper

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

flags.DEFINE_string("env", "FrankaRobotiq-Vision-v0", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 150, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 4, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 100000, "Replay buffer capacity.")

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

id_to_task = {0: "fw", 1: "bw"}
TrainerPortMapping = {
    "fw": {"port_number": 6678, "broadcast_port": 6679},
    "bw": {"port_number": 6690, "broadcast_port": 6691},
}


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


dummy_action = np.array([0.] * 7)

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
                
                obs, _ = env.reset()
                
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
    clients["bw"].recv_network_callback(update_params_bw)

    env.set_task_id(0)
    obs, _ = env.reset(task_id=0)
    obs, reward, done, info = env.step(dummy_action)
    done = False

    current_step = 0
    # training loop
    timer = Timer()
    running_return = 0.0
    start_time = time.time()

    step = {"fw": 0, "bw": 0}

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

    while step["fw"] < FLAGS.max_steps or step["bw"] < FLAGS.max_steps:
        while pause:
            time.sleep(0.1)
        # if restart:
        #     start_time = time.time() - stop_time + start_time
        #     restart = False
        timer.tick("total")
        current_step += 1
        task_name = id_to_task[env.task_id]
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
            # if done:
            #     reward = 1.0
            # else:
            #     reward = 0.0
            if 'pass_for_moving' not in info:
                step[task_name] += 1
                pbars[task_name].update(1)

                # override the action with the intervention action
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

            if 'truncated' in info:
                truncated = True
            else:
                truncated = False

            obs = next_obs
            if done or truncated:
                next_task_id = env.task_id
                print("Final state Reward: ", reward)
                if done:
                    print("task success!")
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
                    "actor/running_return_avg_bw": stats['bw_reward'] / current_step,
                    "actor/step": current_step,
                    "actor/task": env.task_id,
                    "actor/fw_done": fw_done,
                    "actor/bw_done": bw_done,
                    "actor/elapsed_time_sec": elapsed_time,  # 초 단위
                    "actor/elapsed_time_min": elapsed_time / 60,  # 분 단위
                    "actor/elapsed_time_hr": elapsed_time / 3600,  # 시간 단위
                }, step=(step["fw"]+step["bw"]))
                current_step = 0
                
                

                stats = {f"{task_name}_train": info}  # send stats to the learner to log
                stats["env_steps"] = step[task_name]
                done_dict = {0: fw_done, 1: bw_done}
                stats["done"] = done_dict[env.task_id]

                if stats["done"]:
                    temp_stats = {"done": 0}
                    clients[id_to_task[next_task_id]].request("send-stats", temp_stats)

                clients[task_name].request("send-stats", stats)
                running_return = 0.0
                env.set_task_id(next_task_id)
                
                obs, _ = env.reset(task_id=env.task_id)
                

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
    
    done_pause = False
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
        nonlocal done_pause
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            if "env_steps" in payload:
                env_steps = payload["env_steps"]
            wandb_logger.log(payload, step=update_steps)
        if "done" in payload:
            done_pause = payload["done"]
            if done_pause:
                print("\n▶ Task를 성공해 학습 정지됨.")
            
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
        while pause or done_pause:
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
            # agent, update_info = agent.update_high_utd(batch, utd_ratio=8)
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
    if FLAGS.actor:
        env = gym.make("box_picking_camera_env",
                    camera_mode="rgb",
                    max_episode_length=FLAGS.max_traj_length,
                    save_video=True
                    )
        # env = SpacemouseInterventionUR5(env)
    else:
        env = gym.make("box_picking_camera_env",
                    camera_mode="rgb",
                    max_episode_length=FLAGS.max_traj_length,
                    save_video=True,
                    fake_env=True
                    )
        # env = SpacemouseInterventionUR5(env, fake_env=True)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingLIBEROWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraWrapper(env)
    env = GripperPenaltyUR5Wrapper(env, penalty=-0.05)
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
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
            agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=image_keys,
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
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=image_keys,
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
            include_grasp_penalty=True,
        )
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=10000,
            image_keys=image_keys,
            include_grasp_penalty=True,
        )
        import pickle as pkl

        demo_path = FLAGS.demo_path  # 폴더 경로
        if os.path.isdir(demo_path):  # 경로가 폴더인지 확인
            demo_files = sorted([os.path.join(demo_path, f) for f in os.listdir(demo_path) if f.endswith(".pkl")])
        else:
            demo_files = [demo_path]  # 단일 파일인 경우 리스트로 처리
        for demo_file in demo_files:
            with open(demo_file, "rb") as f:
                trajs = pkl.load(f)
                for traj in trajs:
                
                    # if tra['dones']:
                    #     tra['rewards'] = 1.0
                    # else:
                    #     tra['rewards'] = 0.0
                    current_gripper_pos = traj['observations']['state'][0]
                    is_already_open = current_gripper_pos > 0.6  # 완전히 열린 상태 기준 (0.04 근처)
                    is_already_closed = current_gripper_pos < 0.6  # 완전히 닫힌 상태 기준 (0.0 근처)
                    act = traj["actions"][-1]
                    if (act > 0.5 and is_already_closed) or (act < -0.5 and is_already_open):
                        traj["grasp_penalty"] = -0.05
                    else:
                        traj["grasp_penalty"] = 0.0

                    # if tra['grasp_penalty'] < 0:
                    #     tra['grasp_penalty'] = -0.05

                    # if 'instruction' in tra['observations']:
                    #     del tra['observations']['instruction']
                    # if 'instruction' in tra['next_observations']:
                    #     del tra['next_observations']['instruction']
                
                    demo_buffer.insert(traj)
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
