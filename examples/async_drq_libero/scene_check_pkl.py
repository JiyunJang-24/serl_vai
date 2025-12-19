import pickle
import numpy as np
import cv2
import os

def analyze_pickle(file_path):
    """Load and analyze the contents of a pickle file."""
    try:
        scale_factor = 2  # 화면 크기 조절
        with open(file_path, "rb") as f:
            data_list = pickle.load(f)
            for data in data_list:
                img = data['observations']['agentview_image']
                img = np.rot90(img, 2)

                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR 변환

                frame_resized = cv2.resize(
                    frame_bgr, 
                    (frame_bgr.shape[1] * scale_factor, frame_bgr.shape[0] * scale_factor)
                )
                cv2.imshow("Episode Video", frame_resized)
                cv2.waitKey(1)

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
folder_path = "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/bc_demos/grasp_neg"  # 폴더 경로 입력
process_all_pickles(folder_path)
