import datetime
import time
import threading
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from responses import target
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import subprocess

from ur_env.utils.vacuum_gripper import VacuumGripper
from ur_env.utils.rg_gripper import RGGripper
from ur_env.utils.rotations import rotvec_2_quat, quat_2_rotvec, pose_2_rotvec, pose_2_quat

np.set_printoptions(precision=4, suppress=True)


def pos_difference(quat_pose_1: np.ndarray, quat_pose_2: np.ndarray):
    assert quat_pose_1.shape == (7,)
    assert quat_pose_2.shape == (7,)
    p_diff = np.sum(np.abs(quat_pose_1[:3] - quat_pose_2[:3]))

    r_diff = (R.from_quat(quat_pose_1[3:]) * R.from_quat(quat_pose_2[3:]).inv()).magnitude()
    return p_diff + r_diff


class UrImpedanceController(threading.Thread):
    def __init__(
            self,
            robot_ip,
            frequency=100,
            kp=10000,
            kd=2200,
            config=None,
            verbose=False,
            plot=False,
            *args,
            **kwargs
    ):
        super(UrImpedanceController, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        self._is_truncated = threading.Event()
        self.lock = threading.Lock()
        self.async_lock = asyncio.Lock()
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.gripper_timeout = {"timeout": config.GRIPPER_TIMEOUT, "last_grip": time.monotonic() - 1e6}
        self.verbose = verbose
        self.do_plot = plot

        self.target_pos = np.zeros((7,), dtype=np.float32)  # new as quat to avoid +- problems with axis angle repr.
        self.target_grip = np.zeros((1,), dtype=np.float32)
        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force_lowpass = np.zeros((6,), dtype=np.float32)  # force of tool tip
        self.curr_force = np.zeros((6,), dtype=np.float32)

        # self.reset_Q = np.array([np.pi / 2., -np.pi / 2., np.pi / 2., -np.pi / 2., -np.pi / 2., 0.], dtype=np.float32)  # reset state in Joint Space
        self.reset_Q = np.array([-0.10577947298158819, -1.647076269189352, 1.6802166144000452, -1.592215200463766, -1.6467116514789026, -0.22041160265077764], dtype=np.float32)  # reset state in Joint Space
        self.reset_Pose = np.zeros_like(self.reset_Q)
        self.reset_height = np.array([0.5], dtype=np.float32)  # TODO make customizable

        self.delta = config.ERROR_DELTA
        self.fm_damping = config.FORCEMODE_DAMPING
        self.fm_task_frame = config.FORCEMODE_TASK_FRAME
        self.fm_selection_vector = config.FORCEMODE_SELECTION_VECTOR
        self.fm_limits = config.FORCEMODE_LIMITS

        self.ur_control: RTDEControlInterface = None
        self.ur_receive: RTDEReceiveInterface = None
        # import pdb; pdb.set_trace()
        # self._force_release_port(30004)  # 30004 포트 강제 해제
        # self.ur_control = RTDEControlInterface(self.robot_ip)
        # self.ur_receive = RTDEReceiveInterface(self.robot_ip)
        self.robotiq_gripper: RGGripper = None
        self.reconnecting = False  # 중복 재연결 방지 플래그
        self.connected = False
        # only temporary to test
        self.hist_data = [[], []]
        self.horizon = [0, 500]
        self.err = 0
        self.noerr = 0

        # log to file (reset every new run)
        with open("/tmp/console2.txt", 'w') as f:
            f.write("reset\n")
        self.second_console = open("/tmp/console2.txt", 'a')

    def start(self):
        super().start()
        if self.verbose:
            print(f"[RIC] Controller process spawned at {self.native_id}")

    def print(self, msg, both=False):
        self.second_console.write(f'{datetime.datetime.now()} --> {msg}\n')
        if both:
            print(msg)

    async def _force_release_port(self, port):
        """RTDE 포트 (30004) 강제 해제"""
        print(f"[RIC] Releasing port {port}...")
        await self._kill_port_processes(port)
        await self._wait_for_port_release(port)

    async def _kill_port_processes(self, port):
        """포트를 사용하는 모든 프로세스를 강제 종료"""
        try:
            # 리눅스 시스템에서 fuser 명령으로 포트 사용하는 프로세스 강제 종료
            result = subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[RIC] Port {port} processes terminated.")
            else:
                print(f"[RIC] No active processes on port {port}.")
        except Exception as e:
            print(f"[RIC] Error terminating port {port} processes: {e}")

    async def _wait_for_port_release(self, port, timeout=10):
        """포트가 안전하게 해제될 때까지 대기"""
        print(f"[RIC] Waiting for port {port} to be safely released...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.5)
            if not await self._is_port_active(port):
                print(f"[RIC] Port {port} has been safely released.")
                return
        print(f"[RIC] Warning: Port {port} was not released within timeout.")

    async def _is_port_active(self, port):
        """포트가 여전히 사용 중인지 확인"""
        result = subprocess.run(
            f"netstat -anp | grep {port}", shell=True, capture_output=True, text=True
        )
        return str(port) in result.stdout
    
    async def start_ur_interfaces(self, gripper=True):
        import pdb; pdb.set_trace()
        if self.connected:
            print("[RIC] Already connected. Skipping reinitialization.")
            return
        else:
            print("[RIC] Ensuring 30004 port is free...")
            await self._force_release_port(30004)  # 30004 포트 강제 해제
            if self.ur_control:
                self.ur_control.disconnect()
                await asyncio.sleep(0.5)  # 연결 해제 안정 대기
            if self.ur_receive:
                self.ur_receive.disconnect()
                await asyncio.sleep(0.5)  # 연결 해제 안정 대기

            self.ur_control = RTDEControlInterface(self.robot_ip)
            self.ur_receive = RTDEReceiveInterface(self.robot_ip)
            self.connected = True
            if gripper:
                self.robotiq_gripper = RGGripper(gripper="rg2", ip="192.168.0.73", port=502)
                await self.robotiq_gripper.connect()
                await self.robotiq_gripper.activate()
            if self.verbose:
                gr_string = "(with gripper) " if gripper else ""
                print(f"[RIC] Controller connected to robot {gr_string}at: {self.robot_ip}")

    async def restart_ur_interface(self):
        self._is_truncated.set()
        self.print("[RIC] forcemode failed, is now truncated!")

        # disconnect and reconnect, otherwise the controller won't take any commands
        print("[RIC] Ensuring 30004 port is free...")
        await self._force_release_port(30004)  # 30004 포트 강제 해제
        self.ur_control.disconnect()
        try:
            print(f"[RTDE] trying to reconnect")
            self.ur_control.reconnect()
        except RuntimeError:
            self.ur_receive.disconnect()
            for _ in range(10):
                try:
                    self.ur_control.disconnect()
                    self.ur_receive.disconnect()
                    self.connected = False
                    await self.start_ur_interfaces(gripper=False)
                    return
                except Exception as e:
                    print(e)
                    time.sleep(0.2)
    # async def start_ur_interfaces(self, gripper=True):
    #     async with self.async_lock:
    #         if self.connected:
    #             print("[RIC] Already connected. Skipping reinitialization.")
    #             return

    #         print("[RIC] Connecting to UR5...")
    #         try:
    #             # 기존 연결 안전하게 종료
    #             if self.ur_control:
    #                 self.ur_control.disconnect()
    #             if self.ur_receive:
    #                 self.ur_receive.disconnect()

    #             # 새 RTDE 인터페이스 연결
    #             self.ur_control = RTDEControlInterface(self.robot_ip)
    #             self.ur_receive = RTDEReceiveInterface(self.robot_ip)
    #             self.connected = True

    #             # Gripper 연결
    #             if gripper:
    #                 self.robotiq_gripper = RGGripper(gripper="rg2", ip="192.168.0.73", port=502)
    #                 await self.robotiq_gripper.connect()
    #                 await self.robotiq_gripper.activate()

    #             if self.verbose:
    #                 gr_string = "(with gripper) " if gripper else ""
    #                 print(f"[RIC] Controller connected to robot {gr_string}at: {self.robot_ip}")

    #         except Exception as e:
    #             self.connected = False
    #             print(f"[RIC] Connection failed: {e}")

    # async def restart_ur_interface(self):
    #     if self.reconnecting:
    #         print("[RIC] Already reconnecting. Waiting...")
    #         return

    #     async with self.async_lock:
    #         self.reconnecting = True  # 재연결 중 플래그 설정
    #         print("[RIC] Restarting UR5 RTDE Interface...")

    #         # 안전한 종료
    #         if self.ur_control:
    #             self.ur_control.disconnect()
    #         if self.ur_receive:
    #             self.ur_receive.disconnect()

    #         # 최대 10회 재시도
    #         for attempt in range(10):
    #             try:
    #                 await self.start_ur_interfaces(gripper=False)
    #                 self.reconnecting = False
    #                 print(f"[RIC] Successfully reconnected on attempt {attempt + 1}.")
    #                 return
    #             except Exception as e:
    #                 print(f"[RIC] Reconnection attempt {attempt + 1} failed: {e}")
    #                 await asyncio.sleep(0.5)  # 짧은 대기 후 재시도

    #         # 만약 재시도 실패
    #         self.reconnecting = False
    #         self.connected = False
    #         print("[RIC] Failed to reconnect after 10 attempts.")

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def is_moving(self):
        return np.linalg.norm(self.get_state()["vel"], 2) > 0.01

    def set_target_pos(self, target_pos: np.ndarray):
        if target_pos.shape == (7,):
            target_orientation = target_pos[3:]
        elif target_pos.shape == (6,):
            target_orientation = rotvec_2_quat(target_pos[3:])
        else:
            raise ValueError(f"[RIC] target pos has shape {target_pos.shape}")

        with self.lock:
            self.target_pos[:3] = target_pos[:3]
            self.target_pos[3:] = target_orientation

            self.print(f"target: {self.target_pos}")

    def set_reset_Q(self, reset_Q: np.ndarray):
        with self.lock:
            self.reset_Q[:] = reset_Q
        self._reset.set()

    def set_reset_pose(self, reset_pose: np.ndarray):
        with self.lock:
            self.reset_Pose[:] = reset_pose
        self._reset.set()

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = target_grip

    def get_target_pos(self, copy=True):
        with self.lock:
            if copy:
                return self.target_pos.copy()
            else:
                return self.target_pos

    async def _update_robot_state(self):
        pos = self.ur_receive.getActualTCPPose()
        vel = self.ur_receive.getActualTCPSpeed()
        Q = self.ur_receive.getActualQ()
        Qd = self.ur_receive.getActualQd()
        force = self.ur_receive.getActualTCPForce()
        pressure = await self.robotiq_gripper.get_current_pressure()
        obj_status = await self.robotiq_gripper.get_object_status()

        # 3-> no object detected, 0-> sucking empty, [1, 2] obj detected
        grip_status = [-1., 1., 1., 0.][obj_status]

        pressure = pressure if pressure < 99 else 0     # 100 no obj, 99 sucking empty, so they are ignored
        # grip status, 0->neutral, -1->bad (sucking but no obj), 1-> good (sucking and obj)
        grip_status = 1. if pressure > 0 else grip_status
        pressure /= 98.  # pressure between [0, 1]
        with self.lock:
            self.curr_pos[:] = pose_2_quat(pos)
            self.curr_vel[:] = vel
            self.curr_Q[:] = Q
            self.curr_Qd[:] = Qd
            self.curr_force[:] = np.array(force)
            # use moving average (5), since the force fluctuates heavily
            self.curr_force_lowpass[:] = 0.1 * np.array(force) + 0.9 * self.curr_force_lowpass[:]
            self.gripper_state[:] = [pressure, grip_status]

    def get_state(self):
        with self.lock:
            state = {
                "pos": self.curr_pos,
                "vel": self.curr_vel,
                "Q": self.curr_Q,
                "Qd": self.curr_Qd,
                "force": self.curr_force_lowpass[:3],
                "torque": self.curr_force_lowpass[3:],
                "gripper": self.gripper_state
            }
            return state

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def _calculate_force(self):
        target_pos = self.get_target_pos(copy=True)
        with self.lock:
            curr_pos = self.curr_pos
            curr_vel = self.curr_vel

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

    def plot(self):
        if self.horizon[0] < self.horizon[1]:
            self.horizon[0] += 1
            self.hist_data[0].append(self.curr_pos.copy())
            self.hist_data[1].append(self.target_pos.copy())
            return

        self.ur_control.forceModeStop()

        print("[RIC] plotting")
        real_pos = np.array([pose_2_rotvec(q) for q in self.hist_data[0]])
        target_pos = np.array([pose_2_rotvec(q) for q in self.hist_data[1]])

        plt.figure()
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), dpi=200)
        ax_label = [{'x': f'time {self.frequency} [Hz]', 'y': ylabel} for ylabel in ["[mm]", "[rad]"]]
        plot_label = "X Y Z RX RY RZ".split(' ')

        for i in range(6):
            ax = axes[i % 3, i // 3]
            ax.plot(real_pos[:, i], 'b', label=plot_label[i])
            ax.plot(target_pos[:, i], 'g')
            ax.set(xlabel=ax_label[i // 3]['x'], ylabel=ax_label[i // 3]['y'])
            ax.legend()

        fig.suptitle(f"params-->  kp:{self.kp}  kd:{self.kd}")
        plt.show(block=True)
        self.stop()

    async def send_gripper_command(self, force_release=False):
        if force_release:
            await self.robotiq_gripper.automatic_release()
            self.target_grip[0] = 0.0
            return

        timeout_exceeded = (time.monotonic() - self.gripper_timeout["last_grip"]) * 1000 > self.gripper_timeout[
            "timeout"]
        # target grip above threshold and timeout exceeded and not gripping something already
        if self.target_grip[0] > 0.5 and timeout_exceeded and self.gripper_state[1] < 0.5:
            await self.robotiq_gripper.automatic_grip()
            self.target_grip[0] = 0.0
            self.gripper_timeout["last_grip"] = time.monotonic()
            # print("grip")

        # release if below neg threshold and gripper activated (grip_status not zero)
        elif self.target_grip[0] < -0.5 and abs(self.gripper_state[1]) > 0.5:
            await self.robotiq_gripper.automatic_release()
            self.target_grip[0] = 0.0
            # print("release")

    def _truncate_check(self):
        downward_force = self.curr_force_lowpass[2] > 20.
        if downward_force:  # TODO add better criteria
            self._is_truncated.set()
        else:
            self._is_truncated.clear()

    def is_truncated(self):
        return self._is_truncated.is_set()

    def run(self):
        try:
            asyncio.run(self.run_async())  # gripper has to be awaited, both init and commands
        finally:
            self.stop()

    async def _go_to_reset_pose(self):
        self.ur_control.forceModeStop()

        # first disable vaccum gripper
        if self.robotiq_gripper:
            await self.send_gripper_command(force_release=True)
            time.sleep(0.01)

        # then move up (so no boxes are moved)
        success = True
        while self.curr_pos[2] < self.reset_height:
            if self.curr_Q[2] < 0.5:  # if the shoulder joint is near 180deg --> do not move into singularity
                success = success and self.ur_control.speedJ([0., -1., 1., 0., 0., 0.], acceleration=0.8)
                print(f"[RIC] moving up to {self.reset_height} with speedJ (joint space)")
            else:
                success = success and self.ur_control.speedL([0., 0., 0.25, 0., 0., 0.], acceleration=0.8)
                print(f"[RIC] moving up to {self.reset_height} with speedL (task space)")
            await self._update_robot_state()
            time.sleep(0.01)
        self.ur_control.speedStop(a=1.)

        if self.reset_Pose.std() > 0.001:
            success = success and  self.ur_control.moveL(self.reset_Pose, speed=0.5, acceleration=0.3)
            self.print(f"[RIC] moving to {self.reset_Pose} with moveL (task space)", both=self.verbose)
            self.reset_Pose[:] = 0.
        else:
            # then move to desired Jointspace position
            success = success and self.ur_control.moveJ(self.reset_Q, speed=1., acceleration=0.8)
            self.print(f"[RIC] moving to {self.reset_Q} with moveJ (joint space)", both=self.verbose)

        time.sleep(0.1)     # wait for 100ms
        await self._update_robot_state()
        with self.lock:
            self.target_pos = self.curr_pos.copy()

        self.ur_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster
        self.ur_control.zeroFtSensor()

        if not success:     # restart if not successful
            await self.restart_ur_interface()
        else:
            self._reset.clear()

    async def run_async(self):
        import pdb; pdb.set_trace()
        await self.start_ur_interfaces(gripper=True)

        self.ur_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster

        try:
            dt = 1. / self.frequency
            self.ur_control.zeroFtSensor()
            await self._update_robot_state()
            self.target_pos = self.curr_pos.copy()
            print(f"[RIC] target position set to curr pos: {self.target_pos}")

            self._is_ready.set()

            while not self.stopped():
                if self._reset.is_set():
                    await self._update_robot_state()
                    await self._go_to_reset_pose()

                t_now = time.monotonic()

                # update robot state and check for truncation
                await self._update_robot_state()
                self._truncate_check()

                # only used for plotting
                if self.do_plot:
                    self.plot()

                # calculate force
                force = self._calculate_force()
                # print(self.target_pos, self.curr_pos, force)
                self.print(f" p:{self.curr_pos}   f:{self.curr_force_lowpass}   gr:{self.gripper_state}")  # log to file

                # send command to robot
                t_start = self.ur_control.initPeriod()
                fm_successful = self.ur_control.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    force,
                    2,
                    self.fm_limits
                )
                if not fm_successful:  # truncate if the robot ends up in a singularity
                    await self.restart_ur_interface()
                    await self._go_to_reset_pose()

                if self.robotiq_gripper:
                    await self.send_gripper_command()

                self.ur_control.waitPeriod(t_start)

                a = dt - (time.monotonic() - t_now)
                time.sleep(max(0., a))
                self.err, self.noerr = self.err + int(a < 0.), self.noerr + int(a >= 0.)  # some logging
                if a < -0.04:       # log if delay more than 50ms
                    self.print(f"Controller Thread stopped for {(time.monotonic() - t_now)*1e3:.1f} ms")

        finally:
            if self.verbose:
                print(f"[RTDEPositionalController] >dt: {self.err}     <dt (good): {self.noerr}")
            # mandatory cleanup
            self.ur_control.forceModeStop()

            # release gripper
            if self.robotiq_gripper:
                await self.send_gripper_command(force_release=True)
                time.sleep(0.05)

            # move to real home
            pi = 3.1415
            reset_Q = [0, -pi / 2., pi / 2., -pi / 2., -pi / 2., 0.]
            self.ur_control.moveJ(reset_Q, speed=1., acceleration=0.8)

            # terminate
            self.ur_control.disconnect()
            self.ur_receive.disconnect()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}")
