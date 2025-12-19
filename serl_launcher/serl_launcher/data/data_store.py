from threading import Lock
from typing import Union, Iterable

import gym
import jax
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer,
)
import os
from agentlace.data.data_store import DataStoreBase
import numpy as np
from typing import List, Optional, TypeVar

# import oxe_envlogger if it is installed
try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print(
        "rlds logger is not installed, install it if required: "
        "https://github.com/rail-berkeley/oxe_envlogger "
    )
    RLDSLogger = TypeVar("RLDSLogger")


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        **kwargs,
    ):
        MemoryEfficientReplayBuffer.__init__(
            self, observation_space, action_space, capacity, pixel_keys=image_keys, **kwargs
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

    # ensure thread safety
    def insert(self, *args, **kwargs):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore, self).sample(
                *args, **kwargs
            )

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO



def populate_data_store(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    if os.path.isdir(demos_path):
        demo_files = [os.path.join(demos_path, f) for f in os.listdir(demos_path)]
    else:
        demo_files = [demos_path]  # 단일 파일인 경우

    for demo_path in demo_files:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                # transition["observations"] = {
                #     "agentview_image": np.expand_dims(transition["observations"].get("agentview_image"), axis=0)
                # }
                # transition["next_observations"] = {
                #     "agentview_image":  np.expand_dims(transition["next_observations"].get("agentview_image"), axis=0)
                # }
                # transition["observations"] = {"agentview_image": np.expand_dims(transition["observations"].get("agentview_image"), axis=0), "robot0_eye_in_hand_image": np.expand_dims(transition["observations"].get("robot0_eye_in_hand_image"), axis=0),"robot0_eye_in_hand_left_image": np.expand_dims(transition["observations"].get("robot0_eye_in_hand_left_image"), axis=0) }
                
                # transition["next_observations"] = {"agentview_image": np.expand_dims(transition["next_observations"].get("agentview_image"), axis=0), "robot0_eye_in_hand_image": np.expand_dims(transition["next_observations"].get("robot0_eye_in_hand_image"), axis=0), "robot0_eye_in_hand_left_image": np.expand_dims(transition["observations"].get("robot0_eye_in_hand_left_image"), axis=0)}
                
                transition["observations"] = {"robot0_eye_in_hand_image": np.expand_dims(transition["observations"].get("robot0_eye_in_hand_image"), axis=0),"robot0_eye_in_hand_left_image": np.expand_dims(transition["observations"].get("robot0_eye_in_hand_left_image"), axis=0) }
                transition["next_observations"] = {"robot0_eye_in_hand_image": np.expand_dims(transition["next_observations"].get("robot0_eye_in_hand_image"), axis=0), "robot0_eye_in_hand_left_image": np.expand_dims(transition["next_observations"].get("robot0_eye_in_hand_left_image"), axis=0)}
                
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions from {demo_path}.")
    
    return data_store

def populate_data_store_ur5(
    data_store: DataStoreBase,
    demos_path: str,
    key_list: list
):
    """
    Utility function to populate demonstrations data into data_store.
    :return data_store
    """
    import pickle as pkl
    if os.path.isdir(demos_path):
        demo_files = [os.path.join(demos_path, f) for f in os.listdir(demos_path)]
    else:
        demo_files = [demos_path]  # 단일 파일인 경우

    for demo_path in demo_files:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                transition["actions"] = np.array([0, 0, 0, 0, 0, 0, 0])
                transition["observations"] = {
                    key: np.expand_dims(transition["observations"][key], axis=0)
                    for key in key_list if key in transition["observations"]
                }
                transition["next_observations"] = {
                    key: np.expand_dims(transition["next_observations"][key], axis=0)
                    for key in key_list if key in transition["next_observations"]
                }
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions from {demo_path}.")
    
    return data_store

def populate_data_store_with_z_axis_only(
    data_store: DataStoreBase,
    demos_path: str,
):
    """
    Utility function to populate demonstrations data into data_store.
    This will remove the x and y cartesian coordinates from the state.
    :return data_store
    """
    import pickle as pkl
    import numpy as np
    from copy import deepcopy

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                tmp = deepcopy(transition)
                tmp["observations"]["state"] = np.concatenate(
                    (
                        tmp["observations"]["state"][:, :4],
                        tmp["observations"]["state"][:, 6][None, ...],
                        tmp["observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                tmp["next_observations"]["state"] = np.concatenate(
                    (
                        tmp["next_observations"]["state"][:, :4],
                        tmp["next_observations"]["state"][:, 6][None, ...],
                        tmp["next_observations"]["state"][:, 10:],
                    ),
                    axis=-1,
                )
                data_store.insert(tmp)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
