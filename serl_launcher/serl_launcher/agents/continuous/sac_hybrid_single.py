from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet
from typing import Union, List

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import DistilBertTokenizer, FlaxDistilBertModel

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper, InstructionEncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, GraspCritic, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack

task_id_to_str = {0: "fw", 1: "bw"}

class FlaxDistilBERTInstructionEncoder(nn.Module):
    pretrained_name: str = "distilbert-base-uncased"
    output_dim: int = 256
    max_length: int = 32

    def setup(self):
        # 토크나이저는 파라미터 트리에 포함되지 않도록 합니다.
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.pretrained_name)
        # Flax DistilBERT 모델: 인스턴스 생성 후 params 속성으로 파라미터 접근 가능
        self.encoder = FlaxDistilBertModel.from_pretrained(self.pretrained_name)
        self.projection = nn.Dense(self.output_dim)

    def __call__(self, instructions: Union[str, List[str]], train: bool = False):
        if isinstance(instructions, str):
            instructions = [instructions]
        tokenized = self.tokenizer(
            instructions,
            return_tensors="jax",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        # train 모드일 때만 dropout_rng 생성. inference에서는 None으로 전달.
        dropout_rng = self.make_rng('dropout') if train else None
        # deterministic 인자 대신 train 인자를 사용합니다.
        outputs = self.encoder(
            **tokenized,
            params=self.encoder.params,
            dropout_rng=dropout_rng,
            train=train
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # projected = self.projection(cls_embedding)
        return cls_embedding

class SACAgentHybridSingleArm(flax.struct.PyTreeNode):
    """
    Online actor-critic supporting several different algorithms depending on configuration:
     - SAC (default)
     - TD3 (policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1})
     - REDQ (critic_ensemble_size=10, critic_subsample_size=2)
     - SAC-ensemble (critic_ensemble_size>>1)
    
    Compared to SACAgent (in sac.py), this agent has a hybrid policy, with the gripper actions
    learned using DQN. Use this agent for single arm setups.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )
    
    def forward_grasp_critic(
        self,
        observations: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
        task: Optional[int] = None,

    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        py_task = int(task) if task is not None else None
        if py_task is not None:
            name = f"{task_id_to_str[py_task]}_grasp_critic"
        else:
            name = "grasp_critic"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name=name,
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_grasp_critic(
        self,
        observations: Data, 
        rng: PRNGKey,
        task: Optional[int] = None,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_grasp_critic(
            observations, task=task, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy( # type: ignore              
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for temperature Lagrange multiplier.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """
        Forward pass for Lagrange penalty for temperature.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """shared computation between loss functions"""
        batch_size = batch["rewards"].shape[0]

        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        
        next_actions, next_actions_log_probs = next_action_distributions.sample_and_log_prob(seed=rng)
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""
        batch_size = batch["rewards"].shape[0]
        # Extract continuous actions for critic
        actions = batch["actions"][..., :-1]

        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # Evaluate next Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # (critic_ensemble_size, batch_size)

        # Subsample if requested
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # Minimum Q across (subsampled) ensemble members
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        predicted_qs = self.forward_critic(
            batch["observations"], actions, rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info
    

    def grasp_critic_loss_fn(self, batch, task, params: Params, rng: PRNGKey):
        """classes that inherit this class can change this function"""

        batch_size = batch["rewards"].shape[0]
        grasp_action = jnp.round(batch["actions"][..., -1]).astype(jnp.int16) + 1 # Cast env action from [-1, 1] to {0, 1, 2}

         # Evaluate next grasp Qs for all ensemble members (cheap because we're only doing the forward pass)
        target_next_grasp_qs = self.forward_target_grasp_critic(
            batch["next_observations"],
            task=task,
            rng=rng,
        )
        chex.assert_shape(target_next_grasp_qs, (batch_size, 3))

        # Select target next grasp Q based on the gripper action that maximizes the current grasp Q
        next_grasp_qs = self.forward_grasp_critic(
            batch["next_observations"],
            task=task,
            rng=rng,
        )
        # For DQN, select actions using online network, evaluate with target network
        best_next_grasp_action = next_grasp_qs.argmax(axis=-1) 
        chex.assert_shape(best_next_grasp_action, (batch_size,))
        
        target_next_grasp_q = target_next_grasp_qs[jnp.arange(batch_size), best_next_grasp_action]
        chex.assert_shape(target_next_grasp_q, (batch_size,))

        # Compute target Q-values
        grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
        target_grasp_q = (
            grasp_rewards
            + self.config["discount"] * batch["masks"] * target_next_grasp_q
        )
        chex.assert_shape(target_grasp_q, (batch_size,))

        # Forward pass through the online grasp critic to get predicted Q-values
        predicted_grasp_qs = self.forward_grasp_critic(
            batch["observations"],
            task=task, 
            rng=rng, 
            grad_params=params
        )
        chex.assert_shape(predicted_grasp_qs, (batch_size, 3))
        
        # Select the predicted Q-values for the taken grasp actions in the batch
        predicted_grasp_q = predicted_grasp_qs[jnp.arange(batch_size), grasp_action]
        chex.assert_shape(predicted_grasp_q, (batch_size,))
        
        # Compute MSE loss between predicted and target Q-values
        chex.assert_equal_shape([predicted_grasp_q, target_grasp_q])
        grasp_critic_loss = jnp.mean((predicted_grasp_q - target_grasp_q) ** 2)

        info = {
            "grasp_critic_loss": grasp_critic_loss,
            "predicted_grasp_qs": jnp.mean(predicted_grasp_q),
            "target_grasp_qs": jnp.mean(target_grasp_q),
            "grasp_rewards": grasp_rewards.mean(),
        }

        return grasp_critic_loss, info


    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_objective = predicted_q - temperature * log_probs
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        entropy = -next_actions_log_probs.mean()
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
    
    def loss_fns(self, batch, task):
        if task is not None:
            name = f"{task_id_to_str[int(task)]}_grasp_critic"
            return {
                "critic": partial(self.critic_loss_fn, batch),
                name: partial(self.grasp_critic_loss_fn, batch, task),
                "actor": partial(self.policy_loss_fn, batch),
                "temperature": partial(self.temperature_loss_fn, batch),
            }
        else:
            return {
                "critic": partial(self.critic_loss_fn, batch),
                "grasp_critic": partial(self.grasp_critic_loss_fn, batch, task),
                "actor": partial(self.policy_loss_fn, batch),
                "temperature": partial(self.temperature_loss_fn, batch),
            }

    @partial(jax.jit, static_argnames=("task", "pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        task: Optional[int] = None,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "grasp_critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgentHybridSingleArm", dict]:
        """
        Take one gradient step on all (or a subset) of the networks in the agent.

        Parameters:
            batch: Batch of data to use for the update. Should have keys:
                "observations", "actions", "next_observations", "rewards", "masks".
            pmap_axis: Axis to use for pmap (if None, no pmap is used).
            networks_to_update: Names of networks to update (default: all networks).
                For example, in high-UTD settings it's common to update the critic
                many times and only update the actor (and other networks) once.
        Returns:
            Tuple of (new agent, info dict).
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))
        chex.assert_shape(batch["actions"], (batch_size, 7))

        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # Compute gradients and update params
        loss_fns = self.loss_fns(batch, task, **kwargs)
        # Only compute gradients for specified steps
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid gradient steps: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})
        if task is not None:
            key = f"{task_id_to_str[(int(task) + 1) % 2]}_grasp_critic"
            loss_fns[key] = lambda params, rng: (0.0, {})

        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Update target network (if requested)
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # Update RNG
        new_state = new_state.replace(rng=rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("task", "argmax"))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        task: Optional[int] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Sample actions from the policy network, **using an external RNG** (or approximating the argmax by the mode).
        The internal RNG will not be updated.
        """

        dist = self.forward_policy(observations, rng=seed, train=False)
        if argmax:
            ee_actions = dist.mode()
        else:
            ee_actions = dist.sample(seed=seed)
        
        seed, grasp_key = jax.random.split(seed, 2)
        grasp_q_values = self.forward_grasp_critic(observations, task=task, rng=grasp_key, train=False)
        
        # Select grasp actions based on the grasp Q-values
        grasp_action = grasp_q_values.argmax(axis=-1)
        grasp_action = grasp_action - 1 # Mapping back to {-1, 0, 1}

        return jnp.concatenate([ee_actions, grasp_action[..., None]], axis=-1)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "grasp_critic": grasp_critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)

        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[..., :-1]],
            grasp_critic=[observations],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_grasp_divide(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        fw_grasp_critic_def: nn.Module,
        bw_grasp_critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        grasp_critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # Algorithm config
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "fw_grasp_critic": fw_grasp_critic_def,
            "bw_grasp_critic": bw_grasp_critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "fw_grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "bw_grasp_critic": make_optimizer(**grasp_critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)

        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions[..., :-1]],
            fw_grasp_critic=[observations],
            bw_grasp_critic=[observations],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # Config
        assert not entropy_per_dim, "Not implemented"
        if target_entropy is None:
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        
        grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
        grasp_critic_def = partial(
            GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
        )(name="grasp_critic")
        
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            grasp_critic_def=grasp_critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
    

    @classmethod
    def create_pixels_instruction(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs: dict = {
            "hidden_dims": [128, 128],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 4,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        language_keys: Iterable[str] = ("language",),
        augmentation_function: Optional[callable] = None,
        grasp_divide: bool = False,
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "resnet":
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")
        
        encoders[language_keys[0]] = FlaxDistilBERTInstructionEncoder()
        encoder_def = InstructionEncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
            language_keys=language_keys,
        )
        # encoder_def = EncodingWrapper(
        #     encoder=encoders,
        #     use_proprio=use_proprio,
        #     enable_stacking=True,
        #     image_keys=image_keys,
        # )

        
        if grasp_divide:
            encoders = {
                "critic": encoder_def,
                "actor": encoder_def,
                "fw_grasp_critic": encoder_def,
                "bw_grasp_critic": encoder_def,
            }
        else:
            encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
            "grasp_critic": encoder_def,
            }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        if grasp_divide:
            fw_grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
            fw_grasp_critic_def = partial(
                GraspCritic, encoder=encoders["fw_grasp_critic"], network=fw_grasp_critic_backbone
            )(name="fw_grasp_critic")

            bw_grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
            bw_grasp_critic_def = partial(
                GraspCritic, encoder=encoders["bw_grasp_critic"], network=bw_grasp_critic_backbone
            )(name="bw_grasp_critic")
        else:
            grasp_critic_backbone = MLP(**grasp_critic_network_kwargs)
            grasp_critic_def = partial(
                GraspCritic, encoder=encoders["grasp_critic"], network=grasp_critic_backbone
            )(name="grasp_critic")
        
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1]-1,
            **policy_kwargs,
            name="actor",
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )
        if grasp_divide:
            agent = cls.create_grasp_divide(
                rng,
                observations,
                actions,
                actor_def=policy_def,
                critic_def=critic_def,
                fw_grasp_critic_def=fw_grasp_critic_def,
                bw_grasp_critic_def=bw_grasp_critic_def,
                temperature_def=temperature_def,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                image_keys=image_keys,
                augmentation_function=augmentation_function,
                **kwargs,
            )
        else:
            agent = cls.create(
                rng,
                observations,
                actions,
                actor_def=policy_def,
                critic_def=critic_def,
                grasp_critic_def=grasp_critic_def,
                temperature_def=temperature_def,
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                image_keys=image_keys,
                augmentation_function=augmentation_function,
                **kwargs,
            )

        if "pretrained" in encoder_type:  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent

