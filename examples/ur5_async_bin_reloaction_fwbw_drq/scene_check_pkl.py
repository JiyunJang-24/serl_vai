import pickle
import numpy as np
import cv2
import os

def analyze_pickle(file_path):
    """Load and analyze the contents of a pickle file."""
    try:
        scale_factor = 2  # 화면 크기 조절
        # import pdb; pdb.set_trace()  # 디버깅을 위한 중단점
        epi_idx = 0
        steps = 0
        with open(file_path, "rb") as f:
            # import pdb; pdb.set_trace()
            data_list = pickle.load(f)
            # if data_list[0] is not dict:
            #     data_list = data_list[0]

            for idx, data in enumerate(data_list):
                # import pdb; pdb.set_trace()  # 디버깅을 위한 중단점
                print(data["actions"][:3])
                obs = data['observations']
                # if data['rewards'] == 1.0:
                #     import pdb; pdb.set_trace()
                # 'front' 이미지
                front_img = obs['front']
                front_img = cv2.cvtColor(front_img, cv2.COLOR_RGB2BGR)
                front_img = cv2.resize(front_img, (front_img.shape[1] * scale_factor, front_img.shape[0] * scale_factor))

                # 'wrist' 이미지
                # wrist_img = np.rot90(obs['wrist'], 2)
                # wrist_img = obs['wrist']
                # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                # wrist_img = cv2.resize(wrist_img, (wrist_img.shape[1] * scale_factor, wrist_img.shape[0] * scale_factor))

                # 나란히 붙이기 (좌우로)
                combined = np.hstack((front_img, front_img))
                steps += 1
                cv2.imshow("Front (Left) | Wrist (Right)", combined)
                key = cv2.waitKey(5)
                if key == ord('q'):
                    print("Stopped by user")
                    break
                if data['rewards'] == 1.0:
                    print("step: ", steps)
                    print("episode_done!, epi_idx: ", epi_idx)
                    epi_idx += 1
                    steps = 0
    except Exception as e:
        print(f"Error loading pickle file ({file_path}): {e}")

def process_all_pickles(folder_path):
    """Find all pickle files in a folder and process them."""
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    pickle_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    if not pickle_files:
        print(f"No pickle files found in {folder_path}")
        return

    for pkl_file in sorted(pickle_files):  # 정렬된 순서로 파일 처리
        full_path = os.path.join(folder_path, pkl_file)
        print(f"Processing: {full_path}")
        analyze_pickle(full_path)

# 사용 예시
folder_path = "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/Pick_up_the_yellow_tiger_plush_from_the_bin_and_place_it_on_the_brown_plate"  # 폴더 경로 입력
process_all_pickles(folder_path)
