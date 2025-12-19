import pickle
import numpy as np
import cv2
import os

def mark_all_frames_within_each_traj(file_path, output_path, scale_factor=2):
    """Load trajectory file, compute drop location per trajectory, mark all front images in that traj, and save back to .pkl."""
    try:
        with open(file_path, "rb") as f:
            data_list = pickle.load(f)

        # Split into trajectories
        trajectories = []
        current_traj = []
        for data in data_list:
            current_traj.append(data)
            if data['rewards'] == 1.0:
                trajectories.append(current_traj)
                current_traj = []

        updated_data_list = []

        for traj in trajectories:
            drop_frame_idx = None
            for i in range(len(traj) - 1, 0, -1):
                if traj[i]['actions'][-1] < 0:
                    drop_frame_idx = i - 1
                    break

            if drop_frame_idx is not None:
                # 기준이 되는 이미지에서 마커 위치를 구함
                base_obs = traj[drop_frame_idx]['observations']
                front_img_bgr = cv2.cvtColor(base_obs['front'], cv2.COLOR_RGB2BGR)
                hsv = cv2.cvtColor(front_img_bgr, cv2.COLOR_BGR2HSV)
                # lower_purple = np.array([120, 40, 40])
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])

                upper_purple = np.array([160, 255, 255])
                purple_mask = cv2.inRange(hsv, lower_green, upper_green)
                contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 모든 timestep에 대해 마커를 동일하게 적용
                        for step in traj:
                            img_rgb = step['observations']['front']
                            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                            cv2.circle(img_bgr, (cx, cy), 15, (0, 0, 255), thickness=-1)
                            # 다시 RGB로 변환 후 저장
                            step['observations']['front'] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 마킹을 하지 못한 경우 그대로 저장
            updated_data_list.extend(traj)

        # 저장
        with open(output_path, "wb") as f_out:
            pickle.dump(updated_data_list, f_out)

        return output_path

    except Exception as e:
        print(f"Error processing pickle file: {e}")
        return None

# 입력 및 출력 파일 경로
input_pkl = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/place_the_bottle_on_the_white_desk/place_the_bottle_on_the_white_desk_2025_08_01_14_40_13.pkl"
output_pkl = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/place_the_bottle_on_the_white_desk/place_the_bottle_on_the_white_desk_2025_08_01_14_40_13_marked.pkl"

mark_all_frames_within_each_traj(input_pkl, output_pkl)
