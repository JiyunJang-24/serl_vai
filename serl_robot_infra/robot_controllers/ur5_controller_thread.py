import asyncio
import threading
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from ur_env.utils.rg_gripper import RGGripper
from ur_env.utils.vacuum_gripper import VacuumGripper
from ur_env.utils.rotations import rotvec_2_quat, pose_2_quat, pose_2_euler, euler_pose_2_rv, pose_to_tip_pose_rv
import pygame
np.set_printoptions(precision=4, suppress=True)

class SingletonRTDE:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, robot_ip=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, robot_ip):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self.robot_ip = robot_ip
        self.rtde_control = None
        self.rtde_receive = None
        self.robotiq = None
        self.connected = False
        self._conn_lock = asyncio.Lock()

    async def connect(self):
        async with self._conn_lock:
            if self.connected:
                return
            self.rtde_control = RTDEControlInterface(self.robot_ip)
            self.rtde_receive = RTDEReceiveInterface(self.robot_ip)
            self.robotiq = RGGripper(gripper="rg2", ip="192.168.0.72", port=502)
            await self.robotiq.connect()
            await self.robotiq.activate()
            self.connected = True

    async def disconnect(self):
        async with self._conn_lock:
            if self.rtde_control:
                self.rtde_control.disconnect()
            if self.rtde_receive:
                self.rtde_receive.disconnect()
            if self.robotiq:
                await self.robotiq.disconnect()
            self.connected = False

    async def gripper_reconnect(self):
        async with self._conn_lock:
            if self.robotiq:
                await self.robotiq.disconnect()
            self.robotiq = RGGripper(gripper="rg2", ip="192.168.0.72", port=502)
            await self.robotiq.connect()
            await self.robotiq.activate()

    def get_control(self):
        return self.rtde_control

    def get_receive(self):
        return self.rtde_receive

    def get_gripper(self):
        return self.robotiq

class UrImpedanceController_Thread(threading.Thread):
    def __init__(self, robot_ip, config, state_update_callback, frequency=1000, kp=10000, kd=2200, verbose=False, plot=False):
        super().__init__()
        self.rtde = SingletonRTDE(robot_ip)
        self.config = config
        self.state_update_callback = state_update_callback
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.verbose = verbose
        self.do_plot = plot
        # synchronization events
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        # self._is_truncated = threading.Event()
        # self._move_event = threading.Event()
        # state variables
        self.curr_pos = np.zeros(7, dtype=np.float64)
        self.curr_vel = np.zeros(6, dtype=np.float64)
        self.curr_Q = np.zeros(6, dtype=np.float64)
        self.curr_Qd = np.zeros(6, dtype=np.float64)
        self.curr_force_lowpass = np.zeros(6, dtype=np.float64)
        self.curr_force = np.zeros(6, dtype=np.float64)
        self.gripper_state = np.zeros(2, dtype=np.float64)
        self.curr_pos_rv = np.zeros(6, dtype=np.float64)
        self.curr_pos_euler = np.zeros(6, dtype=np.float64)
        # self.target_pos = np.array([-0.3520851312799418, -0.1418549027619048, 0.2251419911572377, -3.0294796774031645, 0.6576746557416007, 0.058019675187175684], dtype=np.float64)
        self.target_pos = None
        self.target_grip = 0.0
        self.reset_Q = None
        self.hist_real = []
        self.hist_target = []
        self.reset_height = 0.5
        self.reset_Pose = np.array([0.19036394544547358, -0.5509824733913188, 0.3743024207434382, 2.1730648280306664, 2.216902529146229, 0.03259730546907828], dtype=np.float64)
        self.reset_joint_Pose = np.array([1.6642441749572754, -1.3628831666759034, 1.6354206244098108, -1.856189867059225, -1.5244134108172815, 1.6480844020843506], dtype=np.float64)
        self.gripper_length = 0.23
        self.gripper_working = True
        self.reset_joint_wrist = np.array([0.0], dtype=np.float64)  # wrist joint angle for reset
    def run(self):
        asyncio.run(self._run_async())

    def stop(self):
        self._stop.set()

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    # def is_truncated(self):
    #     return self._is_truncated.is_set()

    def set_target_pose(self, target_pose):
        assert len(target_pose) in (6, 7)
        if len(target_pose) == 6:
            target_pose_rv = euler_pose_2_rv(target_pose.copy())
            self.target_pos = pose_to_tip_pose_rv(target_pose_rv, -self.gripper_length)
            
        else:
            self.target_pos = target_pose.copy()
            self.target_pos = self.target_pos[:6]
            self.target_pos[3:] = np.array([-3.0294796774031645, 0.6576746557416007, 0.058019675187175684])
        return True
    def set_gripper_pos(self, grip_val):
        self.target_grip = float(grip_val)

    def is_moving(self):
        # Returns True if TCP velocity magnitude exceeds threshold
        return np.linalg.norm(self.curr_vel[:3]) > getattr(self.config, 'MOVING_THRESHOLD', 0.08) or np.linalg.norm(self.curr_vel[3:]) > getattr(self.config, 'MOVING_THRESHOLD', 0.25)

    def set_reset_angles(self, q):
        self.reset_Q = np.array(q, dtype=np.float64)
        self._reset.set()

    def get_state(self):
        return {
            "pos": self.curr_pos.copy(),
            "vel": self.curr_vel.copy(),
            "Q": self.curr_Q.copy(),
            "Qd": self.curr_Qd.copy(),
            "force": self.curr_force_lowpass.copy(),
            "torque": self.curr_force_lowpass.copy()[3:],
            "gripper": self.gripper_state.copy(),
            "pos_rv": self.curr_pos_rv.copy(),
            "pos_euler": self.curr_pos_euler.copy(),
            "gripper_working": self.gripper_working,
        }

    async def _update_robot_state(self, receive):
        pos = receive.getActualTCPPose()
        tip_pos = pose_to_tip_pose_rv(pos, self.gripper_length)
        vel = receive.getActualTCPSpeed()
        Q = receive.getActualQ()
        Qd = receive.getActualQd()
        force = receive.getActualTCPForce()
        gr = self.rtde.get_gripper()
        pressure = await gr.get_current_pressure()
        obj_status = await gr.get_object_status()
        grip_status = [-1., 1., 1., 0.][obj_status]
        pressure = pressure if pressure < 99 else 0
        grip_status = 1. if pressure > 0 else grip_status
        pressure /= 98.
        with threading.Lock():
            self.curr_pos = pose_2_quat(tip_pos) #rv 2 quat
            self.curr_pos_rv = np.array(pos)
            self.curr_pos_euler = pose_2_euler(tip_pos)
            # print(self.curr_pos_euler)
            self.curr_vel = np.array(vel)
            self.curr_Q = np.array(Q)
            # print("Q:", self.curr_Q)
            self.curr_Qd = np.array(Qd)
            self.curr_force = np.array(force)
            self.curr_force_lowpass = 0.1 * self.curr_force + 0.9 * self.curr_force_lowpass
            self.gripper_state = np.array([pressure, grip_status])
            # ← 여기에 콜백 호출
            if self.state_update_callback:
                # sync callback: ur5Env._update_currpos() 실행
                self.state_update_callback()

    def _calculate_force(self):
        target = self.target_pos
        curr_p = self.curr_pos
        curr_v = self.curr_vel
        dp = np.clip(target[:3] - curr_p[:3], -self.config.ERROR_DELTA, self.config.ERROR_DELTA)
        dv = np.clip(-curr_v[:3], -2*self.config.ERROR_DELTA*self.frequency, 2*self.config.ERROR_DELTA*self.frequency)
        force_p = self.kp * dp + self.kd * dv
        rot_diff = R.from_quat(target[3:]) * R.from_quat(curr_p[3:]).inv()
        torque = rot_diff.as_rotvec() * self.config.TORQUE_GAIN
        if self.curr_force[2] > 3.5 and force_p[2] < 0:
            force_p[2] = max((1.5 - self.curr_force_lowpass[2]), 0.) * force_p[2] + min(self.curr_force_lowpass[2] - 0.5, 1.) * 20
        return np.concatenate((force_p, torque))

    async def _run_async(self):
        pygame.init()
        pygame.display.set_mode((400, 400))  # 창을 띄우지 않으면 이벤트 큐가 초기화되지 않음
        pygame.display.set_caption("Controller display")
        await self.rtde.connect()
        control = self.rtde.get_control()
        receive = self.rtde.get_receive()
        gr = self.rtde.get_gripper()
        # original_pos = self.curr_pos_rv.copy()
        # control.forceModeSetDamping(self.config.FORCEMODE_DAMPING)
        # control.zeroFtSensor()
        dt = 1.0 / self.frequency
        init_p = receive.getActualTCPPose()
        init_p_tip = pose_to_tip_pose_rv(init_p)
        self.curr_pos_rv = np.array(init_p)
        self.curr_pos = pose_2_quat(init_p_tip)
        original_pos = self.curr_pos_rv.copy()
        self.target_pos = self.curr_pos_rv.copy()
        self._is_ready.set()
        # print(f"[RIC] Robot is ready, current pose: {self.curr_pos}")
        step = 0
        ts = control.initPeriod()  
        while not self._stop.is_set():
            step+=1
            # print(f"[loop] self id={id(self)}, move_event:{self._move_event.is_set()}")
            if self._reset.is_set():
                # await self._update_robot_state(receive)
                print(f"[RIC] Resetting robot")
                await self._go_to_reset_pose(control, receive, gr)
                # await self._go_to_reset_pose_joint(control, receive, gr)
                # print(f"[RIC] Resetting finished", self._reset.is_set())
                print(f"[RIC] Resetting finished")
                self._reset.clear()

            # if self._move_event.is_set():
            #     await self.go_to_target_pose(control, receive)
            #     print(f"[RIC] Go to target pose")
            #     await self._update_robot_state(receive)
                # 이벤트 클리어
                # 다시 force mode나 다음 루프로
            # print("아무것도 안함")
            # await self._update_robot_state(receive) 

            # 2) 목표 위치로 실시간 streaming
            # print(f"[RIC] servoL: {self.target_pos}")
            await self._update_robot_state(receive)
            force_norm = np.linalg.norm(self.curr_force[:3])
            force = self.curr_force[:3]  # X, Y, Z 방향의 힘
            # if force_norm > 30.0: #for RL
            if force_norm > 60.0: #for imitation
                # print(force_norm, force)
                print("Force Control!!!!")
                # pos_rv = original_pos
                force_norm_z = np.linalg.norm(force[:3])
                dir_vec = force / force_norm_z
                pos_rv = self.curr_pos_rv.copy()
                # import pdb; pdb.set_trace()
                # 접근 방향 계산 (원래 목표 방향)
                # dir_vec = self.target_pos[:3] - pos_rv[:3]
                # norm = np.linalg.norm(dir_vec)
                # if norm > 1e-6:
                #     dir_vec /= norm
                # else:
                #     dir_vec = np.zeros(3)
                # 뒤로 물러날 포즈
                backoff_pose = pos_rv.copy()
                backoff_pose[:3] += dir_vec * 0.02
                # 한 번 servoL 호출해서 물러나기
                control.servoL(
                    backoff_pose.tolist(),
                    0.01, 0.01,
                    0.2, 0.2,
                    100
                )
            else:
            #     print("Force real control!!")
            #     # 상태 갱신 & 루프 대기
            #     await self._update_robot_state(receive)
            #     await asyncio.sleep(dt)
            #     continue

                original_pos = self.curr_pos_rv.copy()
                ok = control.servoL(
                    self.target_pos,                   # List[float] length 6
                    0.01,                     # speed
                    0.01,                     # acceleration
                    0.2,                     # time
                    0.2,                     # lookahead_time
                    100                      # gain
                )
                # ok = control.moveL(
                #     self.target_pos,                   # List[float] length 6
                #     0.25,                     # speed
                #     1.2,                     # acceleration
                #     asynchronous=True,  # Asynchronous mode
                # )
            # print(self.curr_pos_rv)
            # print(self.target_pos)
            # control.moveL(
            #     self.target_pos,                   # List[float] length 6
            #     0.01,                     # speed
            #     0.01,                     # acceleration
            # )
            
            
            # print(self.curr_pos_rv)
            # if self.curr_force[2] > self.config.TRUNCATE_FORCE:
            #     print(f"[RIC] Truncating force: {self.curr_force[2]}")
            #     self._is_truncated.set()
            # else:
            #     print(f"[RIC] Normal force: {self.curr_force[2]}")
            #     self._is_truncated.clear()
            # cmd = self._calculate_force()
            # ts = control.initPeriod()
            # control.forceMode(self.config.FORCEMODE_TASK_FRAME,
            #                   self.config.FORCEMODE_SELECTION_VECTOR,
            #                   cmd, 2, self.config.FORCEMODE_LIMITS)
            # control.waitPeriod(ts)
            # gripper
            cur_grip = self.gripper_state[0]
            if self.target_grip > 0.5:
                if self.gripper_state[0] > 0.6:
                    await gr.automatic_grip()
                self.target_grip = 0.0
            elif self.target_grip < -0.5:
                if self.gripper_state[0] < 0.6:
                    await gr.automatic_release()                    
                self.target_grip = 0.0

            await self._update_robot_state(receive)
            if self.do_plot:
                self.hist_real.append(self.curr_pos.copy())
                self.hist_target.append(self.target_pos.copy())
            # await asyncio.sleep(0.2)

            await asyncio.sleep(dt)
        # control.forceModeStop()
        print("[RIC] Disconnecting RTDE")
        await self.rtde.disconnect()

    def plot(self):
        if not self.do_plot:
            return
        data = np.array(self.hist_real)
        tgt = np.array(self.hist_target)
        fig, axes = plt.subplots(3,2, figsize=(10,6))
        labels = ['X','Y','Z','Rx','Ry','Rz']
        for i in range(6):
            ax = axes[i//2, i%2]
            ax.plot(data[:,i], label='real')
            ax.plot(tgt[:,i], label='target')
            ax.set_title(labels[i])
            ax.legend()
        plt.show()

    async def _go_to_reset_pose(self, control, receive, gr):
        # control.forceModeStop()
        # print(f"[RIC] forceModeStop")
        # first disable vaccum gripper

        # then move up (so no boxes are moved)
        # success = True
        # # while self.curr_pos[2] < self.reset_height:
        # print(f"[RIC] moving up to {self.reset_height}")
        # print("[RIC] current pos: ", self.curr_pos)
        # target_pos = self.curr_pos.copy()
        # target_pos[2] = self.reset_height+0.05
        # control.moveL(target_pos, acceleration=0.8)
        # time.sleep(0.01)
        # print("[RIC] moving up to ", target_pos)
        # await self._update_robot_state()
        # target_pos = self.curr_pos.copy()
        # target_pos[2] = self.reset_height+0.1
        # control.servoL(
        #     target_pos,                   # List[float] length 6
        #     0.01,                     # speed
        #     0.01,                     # acceleration
        #     0.2,                     # time
        #     0.2,                     # lookahead_time
        #     100                      # gain
        # )
        print("[RIC]: Reset Start!!")
        time.sleep(0.5)
        await self._update_robot_state(receive)
        if gr and self.gripper_state[0] < 0.5:
            print(f"[RIC] releasing gripper")
            cur_grip = self.gripper_state[0]
            await gr.automatic_release()
            time.sleep(0.5)
            await self._update_robot_state(receive)
            print(self.gripper_state[0], cur_grip)
            if abs(self.gripper_state[0] - cur_grip) < 0.001:
                print(f"[RIC] Gripper not working")
                self.gripper_working = False
                self.state_update_callback()
                while not self.gripper_working:
                    time.sleep(0.1)
                    while not self.gripper_working:
                        time.sleep(0.1)
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_r:
                                    self.rtde.gripper_reconnect()
                                    print(f"[RIC] Gripper reconnected")
                                    gr = self.rtde.get_gripper()
                                    self.gripper_working = True
                                    self.state_update_callback()
            print(f"[RIC] End of releasing gripper")
        await self._update_robot_state(receive)
        
        # 3) 중간 지점(interpolation) 파라미터
        z_offset = 0.15                    # 먼저 얼마나 올릴지 (m)
        raise_steps = 20                    # 올릴 때 나눌 스텝 수
        reset_steps = 30                   # reset_Pose로 이동할 때 스텝 수
        step_delay = 0.1                   # 각 스텝 사이의 딜레이 (초)

        # 현재 pose, 올릴 pose, 목표 pose (rotvec 기반)
        current_rv  = np.array(self.curr_pos_rv)      # [x,y,z,rvx,rvy,rvz]
        raise_pose  = current_rv.copy()
        raise_pose[2] += z_offset
        target_rv   = np.array(self.reset_Pose)       # [x,y,z,rvx_t, rvy_t, rvz_t]
        # 5) 보간 함수
        def interpolate(start, end, steps):
            for i in range(1, steps + 1):
                alpha = i / steps
                yield (1 - alpha) * start + alpha * end
        print("[RIC]: Raise up")
        # print("current_rv:", current_rv, " raise_pose: ", raise_pose)
        # 6) 중간 포즈 리스트 생성
        for p in interpolate(current_rv, raise_pose, raise_steps):
            # print("interpolation reset control")
            # print(p)
            control.servoL(
                p.tolist(),
                0.02, 0.01, 0.2, 0.2, 100
            )
            await asyncio.sleep(step_delay)
            await self._update_robot_state(receive)

        
        #  # --- 2) Orientation 맞추기 (translation 고정) ---
        ori_only = raise_pose.copy()
        curr_Q = self.curr_Q
        target_Q = curr_Q.copy()
        target_Q[-1] = self.reset_joint_wrist
        control.servoJ(
            target_Q.tolist(),
            0.02, 0.01, 0.2, 0.2, 300                  # List[float] length 6
        )
        # ori_only[3:] = target_rv[3:]
        # control.servoL(
        #     ori_only.tolist(),
        #     0.02, 0.01, 0.2, 0.2, 100
        # )
        await asyncio.sleep(0.3)
        await self._update_robot_state(receive)

        # --- 3) Position 맞추기 (orientation 고정) ---
        # 현 translation, 목표 translation
        trans_start = ori_only[:3]
        trans_end   = target_rv[:3]
        # print("start: ", trans_start, " target: ", trans_end)
        for t in interpolate(trans_start, trans_end, reset_steps):
            p = np.concatenate((t, target_rv[3:]))
            control.servoL(
                p.tolist(),
                0.02, 0.01, 0.2, 0.2, 100
            )
            await asyncio.sleep(step_delay)
            await self._update_robot_state(receive)
        
            time.sleep(0.01)
        
        await self._update_robot_state(receive)
        await asyncio.sleep(0.5)
        await self._update_robot_state(receive)
        self.target_pos = self.curr_pos.copy()
        
        self._reset.clear()


    async def _go_to_reset_pose_joint(self, control, receive, gr):
        
        await self._update_robot_state(receive)
        # 3) 중간 지점(interpolation) 파라미터
        z_offset = 0.15                    # 먼저 얼마나 올릴지 (m)
        raise_steps = 10                    # 올릴 때 나눌 스텝 수
        reset_steps = 15                   # reset_Pose로 이동할 때 스텝 수
        step_delay = 0.1                   # 각 스텝 사이의 딜레이 (초)

        # 현재 pose, 올릴 pose, 목표 pose (rotvec 기반)
        current_rv  = np.array(self.curr_pos_rv)      # [x,y,z,rvx,rvy,rvz]
        raise_pose  = current_rv.copy()
        raise_pose[2] += z_offset
        target_joint   = np.array(self.reset_Pose)       # [x,y,z,rvx_t, rvy_t, rvz_t]
        # 5) 보간 함수
        def interpolate(start, end, steps):
            for i in range(1, steps + 1):
                alpha = i / steps
                yield (1 - alpha) * start + alpha * end

        # 6) 중간 포즈 리스트 생성
        for p in interpolate(current_rv, raise_pose, raise_steps):
            # print("interpolation reset control")
            # print(p)
            # control.servoL(
            #     p.tolist(),
            #     0.02, 0.01, 0.2, 0.2, 100
            # )
            control.moveL(
                p.tolist(),
                speed = 0.25,
                acceleration = 1.2,
                # asynchronous=True,  # Asynchronous mode
            )
            await asyncio.sleep(step_delay)
            await self._update_robot_state(receive)

        if gr and self.gripper_state[0] < 0.65:
            print(f"[RIC] releasing gripper")
            cur_grip = self.gripper_state[0]
            await gr.automatic_release()
            time.sleep(0.1)
            await self._update_robot_state(receive)
            print(self.gripper_state[0], cur_grip)
            if abs(self.gripper_state[0] - cur_grip) < 0.001:
                print(f"[RIC] Gripper not working")
                self.gripper_working = False
                self.state_update_callback()
                while not self.gripper_working:
                    time.sleep(0.1)
                    while not self.gripper_working:
                        time.sleep(0.1)
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_r:
                                    self.rtde.gripper_reconnect()
                                    print(f"[RIC] Gripper reconnected")
                                    gr = self.rtde.get_gripper()
                                    self.gripper_working = True
                                    self.state_update_callback()
            print(f"[RIC] End of releasing gripper")

        # control.speedStop(a=1.) 

        control.servoJ(
            target_joint.tolist(),
                0.02, 0.01, 0.2, 0.2, 300                  # List[float] length 6
        )
        # control.moveJ(
        #     target_joint.tolist(),
        #         1.05, 1.4,                  # List[float] length 6
        #         asynchronous=True,  # Asynchronous mode
        # )
        # for _ in range(100):  # 100 * 0.04s = 4초 이동
        #     control.servoJ(
        #         target_joint.tolist(),
        #         t=0.04,
        #         lookahead_time=0.05,
        #         gain=0.05,
        #         a=0.1,
        #         v=20
        #     )
        #     time.sleep(0.04)
        await asyncio.sleep(step_delay)
        await self._update_robot_state(receive)
        
        # 최종 업데이트
        await asyncio.sleep(0.5)
        await self._update_robot_state(receive)
        self.target_pos = self.curr_pos.copy()
        
        self._reset.clear()

    async def go_to_target_pose(self, control, receive):

        # print(f"[RIC] go_to_target_pose called with target pose: {self.target_pos}")
        control.moveL(self.target_pos, speed=0.3, acceleration=0.5)
        # print(f"[RIC] moveL called with target pose: {self.target_pos}")
        await self._update_robot_state(receive)
        # self.target_pos = self.curr_pos.copy()
        # self._move_event.clear()
        # print(f"[RIC] go_to_target_pose finished")
