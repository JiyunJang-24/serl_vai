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
            self.robotiq = RGGripper(gripper="rg2", ip="192.168.0.75", port=502)
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

    def get_control(self):
        return self.rtde_control

    def get_receive(self):
        return self.rtde_receive

    def get_gripper(self):
        return self.robotiq

class UrImpController_Thread(threading.Thread):
    def __init__(self, robot_ip, config, state_update_callback, frequency=100, kp=3000, kd=400, verbose=False, plot=False):
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
        self.target_pos = np.array([-0.3520851312799418, -0.1418549027619048, 0.2251419911572377, -3.0294796774031645, 0.6576746557416007, 0.058019675187175684], dtype=np.float64)
        self.target_grip = 0.0
        self.reset_Q = None
        self.hist_real = []
        self.hist_target = []
        self.delta = config.ERROR_DELTA if hasattr(config, 'ERROR_DELTA') else 0.05
        self.reset_height = 0.5
        self.reset_Pose = np.array([-0.3520851312799418, -0.1418549027619048, 0.2251419911572377, -3.0294796774031645, 0.6576746557416007, 0.058019675187175684], dtype=np.float64)
        self.gripper_length = 0.23
        self.err = 0
        self.noerr = 0
        self.fm_damping = 0.0  # less damping = Faster
        self.fm_task_frame = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.fm_selection_vector = np.array([1, 1, 1, 1, 1, 1], dtype=np.int8)
        self.fm_limits = np.array([0.5, 0.5, 0.5, 1., 1., 1.], dtype=np.float64)
        self.fm_limits[2] = 0.1  # Z-axis force limit
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
            self.target_pos_quat = pose_2_quat(self.target_pos)
            
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
            self.curr_vel = np.array(vel)
            self.curr_Q = np.array(Q)
            self.curr_Qd = np.array(Qd)
            self.curr_force = np.array(force)
            self.curr_force_lowpass = 0.1 * self.curr_force + 0.9 * self.curr_force_lowpass
            self.gripper_state = np.array([pressure, grip_status])
            # ← 여기에 콜백 호출
            if self.state_update_callback:
                # sync callback: ur5Env._update_currpos() 실행
                self.state_update_callback()

    def _calculate_force(self):
        import pdb; pdb.set_trace()
        target_pos = self.target_pos_quat.copy()
        curr_pos = self.curr_pos.copy()
        curr_vel = self.curr_vel.copy()
        # calc position for
        kp, kd = self.kp, self.kd
        diff_p = np.clip(target_pos[:3] - curr_pos[:3], a_min=-self.delta, a_max=self.delta)
        vel_delta = 2 * self.delta * self.frequency
        diff_d = np.clip(- curr_vel[:3], a_min=-vel_delta, a_max=vel_delta)
        force_pos = kp * diff_p + kd * diff_d

        # calc torque
        rot_diff = R.from_quat(target_pos[3:]) * R.from_quat(curr_pos[3:]).inv()
        vel_rot_diff = R.from_rotvec(curr_vel[3:]).inv()
        torque = rot_diff.as_rotvec() * 100 + vel_rot_diff.as_rotvec() * 22  # TODO make customizable

        # check for big downward tcp force and adapt accordingly
        if self.curr_force[2] > 3.5 and force_pos[2] < 0.:
            force_pos[2] = max((1.5 - self.curr_force_lowpass[2]), 0.) * force_pos[2] + min(self.curr_force_lowpass[2] - 0.5, 1.) * 20.

        return np.concatenate((force_pos, torque))

    async def _run_async(self):
        await self.rtde.connect()
        control = self.rtde.get_control()
        receive = self.rtde.get_receive()
        gr = self.rtde.get_gripper()
        # original_pos = self.curr_pos_rv.copy()
        control.forceModeSetDamping(self.fm_damping)
        control.zeroFtSensor()
        dt = 1.0 / self.frequency
        init_p = receive.getActualTCPPose()
        init_p_tip = pose_to_tip_pose_rv(init_p)
        self.curr_pos_rv = np.array(init_p)
        self.curr_pos = pose_2_quat(init_p_tip)
        original_pos = self.curr_pos_rv.copy()
        self.target_pos = self.curr_pos_rv.copy()
        self.target_pos_quat = self.curr_pos.copy()
        self._is_ready.set()
        # print(f"[RIC] Robot is ready, current pose: {self.curr_pos}")
        step = 0
        ts = control.initPeriod()  
        while not self._stop.is_set():
            step+=1
            # print(f"[loop] self id={id(self)}, move_event:{self._move_event.is_set()}")
            if self._reset.is_set():
                with threading.Lock():
                    print(f"[RIC] forceModeStop")
                    try:
                        control.forceModeStop()
                    except RuntimeError as e:
                        print(f"[WARN] forceModeStop failed: {e}")
                    await asyncio.sleep(0.3)

                    # ✅ RTDE 재연결
                    await self.rtde.disconnect()
                    await asyncio.sleep(0.3)
                    await self.rtde.connect()
                    control = self.rtde.get_control()
                    receive = self.rtde.get_receive()
                    gr = self.rtde.get_gripper()
                    print(f"[RIC] Reconnecting RTDE")
                    await self._update_robot_state(receive)
                    print(f"[RIC] Resetting robot")
                    await self._go_to_reset_pose(control, receive, gr)
                    # print(f"[RIC] Resetting finished", self._reset.is_set())
                    print(f"[RIC] Resetting finished")

                    self._reset.clear()

            # control.forceModeSetDamping(self.config.FORCEMODE_DAMPING)
            else:
                await self._update_robot_state(receive)
                t_now = time.monotonic()

                force = self._calculate_force()
                # print(self.target_pos, self.curr_pos, force)
                # print(f" p:{self.curr_pos}  tp:{self.target_pos_quat} f:{self.curr_force_lowpass}   gr:{self.gripper_state}")  # log to file

                # send command to robot
                fm_successful = control.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    force,
                    2,
                    self.fm_limits
                )
                # if not fm_successful:
                #     print(f"[RIC] Force mode failed, trying to reconnect RTDE")
                #     await self._go_to_reset_pose(control, receive, gr)
                    
                #     time.sleep(10)  
                # wait for a while if forceMode failed
                # if not fm_successful:  # truncate if the robot ends up in a singularity
                #     await self.restart_ur_interface()
                #     await self._go_to_reset_pose()
                # # gripper

                if self.target_grip > 0.5:
                    if self.gripper_state[0] > 0.6:
                        await gr.automatic_grip()
                    self.target_grip = 0.0
                elif self.target_grip < -0.5:
                    if self.gripper_state[0] < 0.6:
                        await gr.automatic_release()
                    self.target_grip = 0.0
                if self.do_plot:
                    self.hist_real.append(self.curr_pos.copy())
                    self.hist_target.append(self.target_pos_quat.copy())
                
                await self._update_robot_state(receive)
                
                a = dt - (time.monotonic() - t_now)
                time.sleep(max(0., a))
                self.err, self.noerr = self.err + int(a < 0.), self.noerr + int(a >= 0.)  # some logging
                if a < -0.04:       # log if delay more than 50ms
                    print(f"Controller Thread stopped for {(time.monotonic() - t_now)*1e3:.1f} ms")
                # await asyncio.sleep(0.2)s

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
       
        
        await self._update_robot_state(receive)
        # 3) 중간 지점(interpolation) 파라미터
        z_offset = 0.1                     # 먼저 얼마나 올릴지 (m)
        raise_steps = 10                    # 올릴 때 나눌 스텝 수
        reset_steps = 15                   # reset_Pose로 이동할 때 스텝 수
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

         # --- 2) Orientation 맞추기 (translation 고정) ---
        ori_only = raise_pose.copy()
        ori_only[3:] = target_rv[3:]
        control.servoL(
            ori_only.tolist(),
            0.02, 0.01, 0.2, 0.2, 100
        )
        await asyncio.sleep(step_delay)
        await self._update_robot_state(receive)

        # --- 3) Position 맞추기 (orientation 고정) ---
        # 현 translation, 목표 translation
        trans_start = ori_only[:3]
        trans_end   = target_rv[:3]
        for t in interpolate(trans_start, trans_end, reset_steps):
            p = np.concatenate((t, target_rv[3:]))
            control.servoL(
                p.tolist(),
                0.02, 0.01, 0.2, 0.2, 100
            )
            await asyncio.sleep(step_delay)
            await self._update_robot_state(receive)
        if gr:
            print(f"[RIC] releasing gripper")
            await gr.automatic_release()
            print(f"[RIC] End of releasing gripper")
            time.sleep(0.01)
        # 최종 업데이트
        await asyncio.sleep(0.5)
        control.forceModeSetDamping(self.fm_damping)  # less damping = Faster
        control.zeroFtSensor()
        await self._update_robot_state(receive)
        self.target_pos = self.curr_pos.copy()
        self.target_pos_quat = self.curr_pos.copy()
        self._reset.clear()

    async def go_to_target_pose(self, control, receive):

        # print(f"[RIC] go_to_target_pose called with target pose: {self.target_pos}")
        control.moveL(self.target_pos, speed=0.3, acceleration=0.5)
        # print(f"[RIC] moveL called with target pose: {self.target_pos}")
        await self._update_robot_state(receive)
        # self.target_pos = self.curr_pos.copy()
        # self._move_event.clear()
        # print(f"[RIC] go_to_target_pose finished")

    async def restart_ur_interface(self, rtde):
            self._reset.is_set()
            print("[RIC] forcemode failed, is now truncated!")
            try:
                print(f"[RTDE] trying to reconnect")
                rtde.disconnect()
                rtde.connect()
            except Exception as e:
                print(f"[RTDE] Reconnect failed: {e}")
                raise e