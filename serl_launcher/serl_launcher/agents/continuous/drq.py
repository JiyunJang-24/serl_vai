import copy
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Optional, Tuple
from typing import Union, List

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from transformers import DistilBertTokenizer, FlaxDistilBertModel
import torch

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper, InstructionEncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack, concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop



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

class DrQAgent(SACAgent):
    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Models
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # Optimizer
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
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
        image_keys: Iterable[str] = ("image",),
    ):
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }

        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
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
            ),
        )

    @classmethod
    def create_drq(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
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
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
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
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
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
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent

    @classmethod
    def create_drq_instruction(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model architecture
        encoder_type: str = "small",
        shared_encoder: bool = True,
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
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
        language_keys: Iterable[str] = ("instruction",),
        **kwargs,
    ):
        """
        Create a new pixel-based agent, with no encoders.
        """

        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        if encoder_type == "small":
            from serl_launcher.vision.small_encoders import SmallEncoder

            encoders = {
                image_key: SmallEncoder(
                    features=(32, 64, 128, 256),
                    kernel_sizes=(3, 3, 3, 3),
                    strides=(2, 2, 2, 2),
                    padding="VALID",
                    pool_method="avg",
                    bottleneck_dim=256,
                    spatial_block_size=8,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet":
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

        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # Define networks
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
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
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            **kwargs,
        )

        if encoder_type == "resnet-pretrained":  # load pretrained weights for ResNet-10
            from serl_launcher.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)

        return agent
    
    def data_augmentation_fn(self, rng, observations):
        for pixel_key in self.config["image_keys"]:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
    def update_high_utd(
        self,
        batch: Batch,
        *,
        utd_ratio: int,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        """
        Fast JITted high-UTD version of `.update`.

        Splits the batch into minibatches, performs `utd_ratio` critic
        (and target) updates, and then one actor/temperature update.

        Batch dimension must be divisible by `utd_ratio`.

        It also performs data augmentation on the observations and next_observations
        before updating the network.
        """
        new_agent = self
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

        new_state = self.state.replace(rng=rng)

        new_agent = self.replace(state=new_state)
        return SACAgent.update_high_utd(
            new_agent, batch, utd_ratio=utd_ratio, pmap_axis=pmap_axis
        )

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update_critics(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
    ) -> Tuple["DrQAgent", dict]:
        new_agent = self
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)

        rng = new_agent.state.rng
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = self.data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = self.data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )

        new_state = self.state.replace(rng=rng)
        new_agent = self.replace(state=new_state)
        new_agent, critic_infos = new_agent.update(
            batch,
            pmap_axis=pmap_axis,
            networks_to_update=frozenset({"critic"}),
        )
        del critic_infos["actor"]
        del critic_infos["temperature"]

        return new_agent, critic_infos
