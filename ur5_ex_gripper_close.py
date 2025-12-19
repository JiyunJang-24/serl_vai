import rtde_control
import rtde_receive
import time
import threading
import numpy as np
from onrobot_rg.src.onrobot import RG
from scipy.spatial.transform import Rotation as R

# UR5 IP 주소 설정
UR5_IP = "192.168.0.9"
RG2_IP = "192.168.0.73"
RG2_PORT = 502
# RTDE Control 및 Receive 인터페이스 초기화
rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
rg = RG('rg2', RG2_IP, RG2_PORT)
current_pose = rtde_r.getActualTCPPose()
print(current_pose)
def pose_to_tip_pose_rv(rotvec_pose, gripper_length=0.23):
    """
    Converts UR5e EEF pose (pos + rotvec) to tip pose (pos + rotvec)
    Args:
        eef_pose: [x, y, z, rx, ry, rz]
        gripper_length: scalar (in meters)
    Returns:
        tip_pose: [tip_x, tip_y, tip_z, tip_rx, tip_ry, tip_rz]
    """
    position = np.array(rotvec_pose[:3])
    rotvec = np.array(rotvec_pose[3:])
    rotation = R.from_rotvec(rotvec)

    # Gripper offset along local z-axis
    offset = np.array([0, 0, gripper_length])
    rotated_offset = rotation.apply(offset)

    tip_position = position + rotated_offset
    tip_rotvec = rotvec  # Orientation is same as EEF

    tip_pose = np.concatenate([tip_position, tip_rotvec])
    return tip_pose

# print("현재 UR5 TCP 위치:", current_pose)
# current_tip_pose = pose_to_tip_pose_rv(current_pose)
# print("현재 UR5 TIP 위치:", current_tip_pose)
# current_pose[3] += 0.1
# rtde_c.moveL(current_pose, 0.1, 0.1)
# rg.close_gripper(force_val=400)
# 이동 명령 실행
# rtde_c.moveL([-0.42558491447779007, -0.10287609890500561, 0.1778930167385912, 3.087378462920458, -0.4113399802184695, 0.019699037118545547], 0.1, 0.1)
# # rg.open_gripper()
# rtde_c.moveL([-0.40, 0.0972, 0.403, -2.511240650372864, 1.8876180139797094, -6.425309051307278e-05], 0.1, 0.1)
# # rg.close_gripper()
# rtde_c.moveL([-0.35, 0.0672, 0.423, -2.511240650372864, 1.8876180139797094, -6.425309051307278e-05], 0.1, 0.1)
# rg.open_gripper()

print(rg.get_status)
time.sleep(1)  # 이동 후 잠시 대기
