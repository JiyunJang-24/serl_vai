import rtde_control
import rtde_receive
import time
from onrobot_rg.src.onrobot import RG
# UR5 IP 주소 설정
UR5_IP = "192.168.0.9"
RG2_IP = "192.168.0.73"
RG2_PORT = 502
# RTDE Control 및 Receive 인터페이스 초기화
# rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
# rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
rg = RG('rg2', RG2_IP, RG2_PORT)
# current_pose = rtde_r.getActualTCPPose()
# print("현재 UR5 TCP 위치:", current_pose)
# rg.close_gripper(force_val=400)
# 이동 명령 실행
# rtde_c.moveL([-0.35, 0.0472, 0.423, -2.511240650372864, 1.8876180139797094, -6.425309051307278e-05], 0.1, 0.1)
# # rg.open_gripper()
# rtde_c.moveL([-0.40, 0.0972, 0.403, -2.511240650372864, 1.8876180139797094, -6.425309051307278e-05], 0.1, 0.1)
# # rg.close_gripper()
# rtde_c.moveL([-0.35, 0.0672, 0.423, -2.511240650372864, 1.8876180139797094, -6.425309051307278e-05], 0.1, 0.1)
rg.open_gripper()

print(rg.get_status)
time.sleep(1)  # 이동 후 잠시 대기
