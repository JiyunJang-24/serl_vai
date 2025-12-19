import asyncio
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import rtde_control
import rtde_receive
from onrobot_rg.src.onrobot import RG

# ============ IP 설정 ============
UR5_IP = "192.168.0.9"
RG2_IP = "192.168.0.73"
RG2_PORT = 502

# ============ RTDE 인터페이스 ============
rtde_c = rtde_control.RTDEControlInterface(UR5_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(UR5_IP)
rg = RG('rg2', RG2_IP, RG2_PORT)
target_serial = "109622073973"
# ============ 상태 변수 ============
current_pose = None
current_force = None
current_velocity = None

# ============ 카메라 설정 ============
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(target_serial)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# === RealSense 내재 파라미터 (예시) ===
# K = np.array([
#     [615.0, 0, 320.0],
#     [0, 615.0, 240.0],
#     [0,   0,    1.0]
# ], dtype=np.float32)

def get_intrinsics_from_frame(frame):
    profile = frame.profile.as_video_stream_profile()
    intr = profile.get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]])
    return K

# === Extrinsic transform (예: 로봇 base → 카메라 좌표계, 직접 측정 필요) ===
# 여기선 단순화를 위해 identity로 처리 (실제 환경에 맞게 보정 필요)
T_robot_to_cam = np.eye(4)

# ============ Pose 처리 유틸 ============
def pose_to_tip_pose_rv(rotvec_pose, gripper_length=0.23):
    position = np.array(rotvec_pose[:3])
    rotvec = np.array(rotvec_pose[3:])
    rotation = R.from_rotvec(rotvec)
    offset = np.array([0, 0, gripper_length])
    rotated_offset = rotation.apply(offset)
    tip_position = position + rotated_offset
    tip_rotvec = rotvec  # orientation 동일
    tip_pose = np.concatenate([tip_position, tip_rotvec])
    return tip_pose

def draw_axis(img, rvec, tvec, K, scale=0.05):
    axis = np.float32([
        [scale, 0, 0],  # X-red
        [0, scale, 0],  # Y-green
        [0, 0, scale]   # Z-blue
    ])
    origin = np.zeros((1, 3), dtype=np.float32)

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, None)
    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, K, None)
    origin_2d = tuple(origin_2d[0].ravel().astype(int))

    img = cv2.line(img, origin_2d, tuple(imgpts[0].ravel().astype(int)), (0,0,255), 3)
    img = cv2.line(img, origin_2d, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 3)
    img = cv2.line(img, origin_2d, tuple(imgpts[2].ravel().astype(int)), (255,0,0), 3)
    return img

def draw_rotation_vector_direction(img, rotvec, scale=100):
    """
    시각적으로 rx, ry는 방향 벡터, rz는 회전 방향과 세기 표현
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # === 1. x/y 방향 화살표 (빨간색) ===
    direction = rotvec[:2] * scale
    direction = direction.astype(int)
    arrow_tip = (center[0] + direction[0], center[1] - direction[1])
    img = cv2.arrowedLine(img, center, arrow_tip, (0, 0, 255), 2, tipLength=0.2)

    # === 2. z축 회전 표현 (파란색 원호) ===
    rz = rotvec[2]
    if np.abs(rz) > 1e-3:  # 무시할 만큼 작지 않다면
        # 회전 반경, 회전 각도
        radius = 40
        max_angle = 360  # 최대 시각화 각도
        clamped_rz = np.clip(rz, -2.0, 2.0)  # 너무 커지지 않도록
        arc_angle = int(np.abs(clamped_rz) / 2.0 * max_angle)  # rz=2이면 360도

        # 방향: 반시계 (rz>0) or 시계 (rz<0)
        if rz > 0:
            start_angle = 0
            end_angle = arc_angle
        else:
            start_angle = 0
            end_angle = -arc_angle

        # 원호 그리기
        color = (255, 0, 0)
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, 2)

        # 끝점에 점 표시
        angle_deg = end_angle
        angle_rad = np.deg2rad(angle_deg)
        tip_x = int(center[0] + radius * np.cos(angle_rad))
        tip_y = int(center[1] - radius * np.sin(angle_rad))
        img = cv2.circle(img, (tip_x, tip_y), 4, color, -1)

    return img

# ============ 비동기 로봇 상태 수신 ============
async def receive_robot_state(frequency=100):
    global current_pose, current_force, current_velocity
    dt = 1 / frequency
    while True:
        raw_pose = rtde_r.getActualTCPPose()
        current_pose = pose_to_tip_pose_rv(raw_pose)
        current_force = rtde_r.getActualTCPForce()
        current_velocity = rtde_r.getActualTCPSpeed()
        await asyncio.sleep(dt)

# ============ 시각화 루프 ============
async def visualize_loop(frequency=30):
    dt = 1 / frequency
    while True:
        # 1. 카메라 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        K = get_intrinsics_from_frame(color_frame)
        # 2. 로봇 pose가 있으면 시각화
        if current_pose is not None:
            # tvec = np.array(current_pose[:3], dtype=np.float32).reshape(3, 1)
            rvec = np.array(current_pose[3:], dtype=np.float32).reshape(3, 1)

            # Transform pose from robot → camera 좌표계 (단순화됨)
            # T_robot_to_cam 적용 생략 (필요시 적용)
            # color_image = draw_axis(color_image, rvec, tvec, K, scale=0.05)
            color_image = draw_rotation_vector_direction(color_image, rvec.flatten(), scale=10)

        # 3. 시각화
        cv2.imshow("RealSense + Robot Pose", color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        await asyncio.sleep(dt)

# ============ 메인 ============
async def main():
    await asyncio.gather(
        receive_robot_state(frequency=100),
        visualize_loop(frequency=30)
    )

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("종료 중...")
    pipeline.stop()
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("완료.")
