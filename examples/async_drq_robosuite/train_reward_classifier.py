import pickle as pkl
import jax
from jax import numpy as jnp
import flax
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
import gym
import os
from absl import app, flags
import numpy as np
from serl_launcher.wrappers.chunking import ChunkingWrapper, ChunkingLIBEROWrapper
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    populate_data_store,
)
from serl_launcher.networks.reward_classifier import create_classifier

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, SERLObsLIBEROWrapper
from serl_launcher.wrappers.front_camera_wrapper import FrontCameraWrapper, FrontCameraLIBEROWrapper, FWBWFrontCameraRewardClassifierWrapper, FWBWFrontCameraBinaryRewardClassifierWrapper
# from franka_env.envs.relative_env import RelativeFrame
# from franka_env.envs.wrappers import (
#     SpacemouseIntervention,
#     Quat2EulerWrapper,
#     FWBWFrontCameraBinaryRewardClassifierWrapper,
# )
xyz_bounding_box = np.array([[0.005, -0.14, 0.83], [0.19, 0.14, 0.89]])

import sys
import os

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, OnScreenRenderEnv
from robosuite.wrappers import VisualizationWrapper

# Set above env export to prevent OOM errors from memory preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("positive_demo_paths", None, "paths to positive demos")
flags.DEFINE_multi_string("negative_demo_paths", None, "paths to negative demos")
flags.DEFINE_string("classifier_ckpt_path", ".", "Path to classifier checkpoint")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_integer("num_epochs", 100, "Number of epochs for training")


def main(_):
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

    # env = RelativeFrame(env)
    # env = Quat2EulerWrapper(env)
    # env = SERLObsWrapper(env)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = FrontCameraLIBEROWrapper(env)

    # we will only use the front camera view for training the reward classifier
    train_reward_classifier(env.front_observation_space, env.action_space)


def train_reward_classifier(observation_space, action_space):
    """
    User can provide custom observation space to be used as the
    input to the classifier. This function is used to train a reward
    classifier using the provided positive and negative demonstrations.

    NOTE: this function is duplicated and used in both
    async_bin_relocation_fwbw_drq and async_cable_route_drq examples
    """
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    image_keys = [k for k in observation_space.keys() if "state" not in k]
    pos_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=50000,
        image_keys=image_keys,
    )
    pos_buffer = populate_data_store(pos_buffer, FLAGS.positive_demo_paths[0])

    neg_buffer = MemoryEfficientReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=50000,
        image_keys=image_keys,
    )
    neg_buffer = populate_data_store(neg_buffer, FLAGS.negative_demo_paths[0])

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["next_observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    # Define the training step
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["data"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    # Training Loop
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        sample = concat_batches(
            pos_sample["next_observations"], neg_sample["observations"], axis=0
        )
        rng, key = jax.random.split(rng)
        # sample = data_augmentation_fn(key, sample)
        labels = jnp.concatenate(
            [
                jnp.ones((FLAGS.batch_size // 2, 1)),
                jnp.zeros((FLAGS.batch_size // 2, 1)),
            ],
            axis=0,
        )
        batch = {"data": sample, "labels": labels}
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
    # this is used to save the without the orbax checkpointing
    flax.config.update("flax_use_orbax_checkpointing", False)
    rng, key = jax.random.split(rng)
    test_sample = next(pos_iterator)["next_observations"]
    original_logits = classifier.apply_fn({"params": classifier.params}, test_sample, train=False, rngs={"dropout": key})

    print("Original classifier logits before saving:")
    print(original_logits)
    checkpoints.save_checkpoint(
        FLAGS.classifier_ckpt_path,
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )

if __name__ == "__main__":
    app.run(main)
