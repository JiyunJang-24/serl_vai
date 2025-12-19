import pickle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def visualize_last_drop_per_trajectory_with_metadata(file_path, scale_factor=2):
    """Visualize drop moments with trajectory index and step index metadata."""
    marked_images = []
    metadata = []

    try:
        with open(file_path, "rb") as f:
            data_list = pickle.load(f)

        # Split into trajectories based on reward == 1.0
        trajectories = []
        current_traj = []
        for data in data_list:
            current_traj.append(data)
            if data['rewards'] == 1.0:
                trajectories.append(current_traj)
                current_traj = []

        for traj_idx, traj in enumerate(trajectories):
            # Walk backward from end to find last drop step
            drop_frame_idx = None
            for i in range(len(traj) - 1, 0, -1):
                # if traj[i]['actions'][-1] >= 0 and traj[i - 1]['actions'][-1] < 0:
                if traj[i]['actions'][-1] < 0:
                    drop_frame_idx = i - 1
                    break

            if drop_frame_idx is not None:
                obs = traj[drop_frame_idx]['observations']
                front_img = obs['front']

                # Convert and resize image
                front_img_bgr = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                front_img_bgr = cv2.resize(front_img_bgr, (front_img_bgr.shape[1] * scale_factor,
                                                           front_img_bgr.shape[0] * scale_factor))

                # HSV로 보라색 마스크 추출
                hsv = cv2.cvtColor(front_img_bgr, cv2.COLOR_BGR2HSV)
                lower_purple = np.array([120, 40, 40])
                upper_purple = np.array([160, 255, 255])
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])
                purple_mask = cv2.inRange(hsv, lower_green, upper_green)

                # 보라색 영역의 contour 중심 찾기
                contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(front_img_bgr, (cx, cy), 15, (0, 0, 255), thickness=-1)

                marked_images.append(cv2.cvtColor(front_img_bgr, cv2.COLOR_BGR2RGB))
                metadata.append((traj_idx, drop_frame_idx))

        return marked_images, metadata

    except Exception as e:
        print(f"Error loading or processing pickle file ({file_path}): {e}")
        return [], []

# 파일 경로는 실제 환경에서 유효한 것으로 교체해야 함
demo_file = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/place_the_bottle_on_the_white_desk/place_the_bottle_on_the_white_desk_3_2025_08_01_12_56_57.pkl"

marked_imgs, metadata = visualize_last_drop_per_trajectory_with_metadata(demo_file)

# 시각화
for i, (img, (traj_idx, step_idx)) in enumerate(zip(marked_imgs, metadata)):
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Trajectory #{traj_idx}, Drop Step #{step_idx}")
    plt.axis('off')
    plt.show()
