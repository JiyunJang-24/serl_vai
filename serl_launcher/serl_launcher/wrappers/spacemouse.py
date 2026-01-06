import threading
import pyspacemouse
import numpy as np
from typing import Tuple
import time
import gym
import sys

class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]



class SpaceMouseLIBEROExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                # self.latest_data["action"] = np.array(
                #     [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                # )  # spacemouse axis matched with robot base frame
                self.latest_data["action"] = np.array(
                    [-state.y, -state.x, state.z, state.roll, -state.pitch, state.yaw],
                    dtype=np.float64
                )
                self.latest_data["action"][:3] *= 0.75
                self.latest_data["action"][3:] *= 0.25
                
                for i in range(6):
                    if abs(self.latest_data["action"][i]) < 0.1:
                        self.latest_data["action"][i] = 0
                self.latest_data["buttons"] = state.buttons
            time.sleep(0.001)
            
    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]



class SpacemouseInterventionLIBERO(gym.ActionWrapper):
    def __init__(self, env):
        self.env = env
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseLIBEROExpertV2()
        self.last_intervene = 0
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)

        if np.linalg.norm(expert_a) > 0.001:
            self.last_intervene = time.time()
        else:
            expert_a[:] = 0

        if self.gripper_enabled:
            if self.left:  # open gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                self.last_intervene = time.time()
            elif self.right:  # close gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                self.last_intervene = time.time()
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.01:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)        
        # new_action = np.clip(new_action, self.env.action_space.low, self.env.action_space.high)
        obs, rew, done, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, info
    
    def render(self):
        return self.env.render()
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class SpaceMouseLIBEROExpertV2:
    """
    SpaceMouse에서 액션을 읽고 제공하는 클래스 (끊어졌을 때만 자동 재연결)
    """

    def __init__(self):
        pyspacemouse.open()
        self.connected = True
        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6, dtype=np.float64), "buttons": [0, 0]}
        self.last_update = time.time()  # SpaceMouse 업데이트 시간 기록

        self.thread = threading.Thread(target=self._spacemouse_loop, daemon=True)
        self.thread.start()

    def _connect_spacemouse(self):
        """SpaceMouse 연결을 시도하고 성공 여부를 반환"""
        if not self.connected:  # **✅ 이미 연결되어 있으면 다시 실행하지 않음**
            try:
                pyspacemouse.open()
                self.connected = True
                print("✅ SpaceMouse 연결 성공!")
            except Exception as e:
                self.connected = False
                print(f"❌ SpaceMouse 연결 실패: {e}")

    def _spacemouse_loop(self):
        """SpaceMouse 입력을 지속적으로 읽는 루프 (끊어졌을 때만 재연결)"""
        while True:
            try:
                state = pyspacemouse.read()  # ✅ 연결이 끊기면 예외 발생
                time.sleep(0.001)  # **🔹 1ms 대기**
                with self.state_lock:
                    # **✅ np.float64로 변환하여 오류 방지**
                    
                    new_action = np.array(
                        [-state.x, state.y, state.z, state.roll, -state.pitch, state.yaw], 
                        dtype=np.float64
                    )

                    # 이동 감쇠
                    new_action[:3] *= 1.0
                    new_action[4] *= 0.1
                    new_action[3] *= 0.25
                    new_action[5] *= 0.25

                    # **📌 SpaceMouse가 멈추면 자동으로 0으로 설정**
                    for i in range(6):
                        if abs(self.latest_data["action"][i]) < 0.1:
                            self.latest_data["action"][i] = 0

                    self.latest_data["action"] = new_action
                    self.latest_data["buttons"] = state.buttons

                # time.sleep(0.02)  # **🔹 호출 빈도를 줄여 시스템 과부하 방지**

            except Exception as e:
                print(f"❌ SpaceMouse 읽기 오류 발생! 연결이 끊겼음: {e}")
                self.connected = False  # ✅ 연결이 끊기면 다시 재연결하도록 설정
                if not self.connected:
                    self._connect_spacemouse()
                    time.sleep(2)  # **🔹 재연결 시 2초 대기 (CPU 과부하 방지)**

    def get_action(self) -> Tuple[np.ndarray, list]:
        """현재 SpaceMouse 액션 반환 (마지막 입력 이후 0.3초가 지나면 자동 정지)"""
        return self.latest_data["action"], self.latest_data["buttons"]
    

class SpacemouseInterventionUR5(gym.ActionWrapper):
    def __init__(self, env, fake_env=False):
        self.env = env
        self.gripper_enabled = True
        
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        if self.action_space.shape == (4,):
            self.only_pos_control = True
        else:
            self.only_pos_control = False
        if fake_env:
            print("Using Fake SpaceMouse Expert")
            return
        self.expert = SpaceMouseUR5Expert(only_pos_control=self.only_pos_control)
        self.last_intervene = 0
        self.left, self.right = False, False

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)

        if np.linalg.norm(expert_a) > 0.001:
            self.last_intervene = time.time()
        else:
            expert_a[:] = 0

        if self.gripper_enabled:
            if self.left:  # open gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                self.last_intervene = time.time()
            elif self.right:  # close gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                self.last_intervene = time.time()
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.01:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)        
        # new_action = np.clip(new_action, self.env.action_space.low, self.env.action_space.high)
        obs, rew, terminated, truncated,info = self.env.step(new_action)
        done = terminated or truncated
        if replaced:
            # print(new_action)
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, info
    
    def render(self):
        return self.env.render()
    
    def get_state_obs(self):
        return self.env.get_state_obs()
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

class SpaceMouseUR5Expert:
    """
    SpaceMouse에서 액션을 읽고 제공하는 클래스 (끊어졌을 때만 자동 재연결)
    """

    def __init__(self, only_pos_control):
        pyspacemouse.open()
        self.connected = True
        self.only_pos_control = only_pos_control
        # self.state_lock = threading.Lock()
        if self.only_pos_control:
            self.latest_data = {"action": np.zeros(3, dtype=np.float64), "buttons": [0, 0]}
            self.thread = threading.Thread(target=self._spacemouse_loop_only_pos, daemon=True)
        else:
            self.latest_data = {"action": np.zeros(6, dtype=np.float64), "buttons": [0, 0]}
            self.thread = threading.Thread(target=self._spacemouse_loop, daemon=True)
        self.last_update = time.time()  # SpaceMouse 업데이트 시간 기록

        self.thread.start()

    def _connect_spacemouse(self):
        """SpaceMouse 연결을 시도하고 성공 여부를 반환"""
        if not self.connected:  # **✅ 이미 연결되어 있으면 다시 실행하지 않음**
            try:
                pyspacemouse.open()
                self.connected = True
                print("✅ SpaceMouse 연결 성공!")
            except Exception as e:
                self.connected = False
                print(f"❌ SpaceMouse 연결 실패: {e}")

    def _spacemouse_loop(self):
        """SpaceMouse 입력을 지속적으로 읽는 루프 (끊어졌을 때만 재연결)"""
        while True:
            try:
                state = pyspacemouse.read()  # ✅ 연결이 끊기면 예외 발생
                # time.sleep(0.001)  # **🔹 1ms 대기**
                # with self.state_lock:
                    # **✅ np.float64로 변환하여 오류 방지**
                
                new_action = np.array(
                    [-state.x, -state.y, state.z, -state.pitch, state.roll, -state.yaw], 
                    dtype=np.float64
                )

                # **📌 SpaceMouse가 멈추면 자동으로 0으로 설정**
                for i in range(6):
                    if abs(self.latest_data["action"][i]) < 0.1:
                        self.latest_data["action"][i] = 0

                self.latest_data["action"] = new_action
                self.latest_data["buttons"] = state.buttons

                # time.sleep(0.05)  # **🔹 호출 빈도를 줄여 시스템 과부하 방지**

            except Exception as e:
                print(f"❌ SpaceMouse 읽기 오류 발생! 연결이 끊겼음: {e}")
                self.connected = False  # ✅ 연결이 끊기면 다시 재연결하도록 설정
                if not self.connected:
                    self._connect_spacemouse()
                    time.sleep(2)  # **🔹 재연결 시 2초 대기 (CPU 과부하 방지)**

    def _spacemouse_loop_only_pos(self):
        """SpaceMouse 입력을 지속적으로 읽는 루프 (끊어졌을 때만 재연결)"""
        while True:
            try:
                state = pyspacemouse.read()  # ✅ 연결이 끊기면 예외 발생
                # time.sleep(0.001)  # **🔹 1ms 대기**
                # with self.state_lock:
                    # **✅ np.float64로 변환하여 오류 방지**
                
                new_action = np.array(
                    [-state.y, state.x, state.z], 
                    dtype=np.float64
                )

                # 이동 감쇠
                new_action[:3] *= 1.0

                # **📌 SpaceMouse가 멈추면 자동으로 0으로 설정**
                for i in range(3):
                    if abs(self.latest_data["action"][i]) < 0.1:
                        self.latest_data["action"][i] = 0

                self.latest_data["action"] = new_action
                self.latest_data["buttons"] = state.buttons

                time.sleep(0.0005)  # **🔹 호출 빈도를 줄여 시스템 과부하 방지**

            except Exception as e:
                print(f"❌ SpaceMouse 읽기 오류 발생! 연결이 끊겼음: {e}")
                self.connected = False  # ✅ 연결이 끊기면 다시 재연결하도록 설정
                if not self.connected:
                    self._connect_spacemouse()
                    time.sleep(2)  # **🔹 재연결 시 2초 대기 (CPU 과부하 방지)**

    def get_action(self) -> Tuple[np.ndarray, list]:
        """현재 SpaceMouse 액션 반환 (마지막 입력 이후 0.3초가 지나면 자동 정지)"""
        return self.latest_data["action"], self.latest_data["buttons"]