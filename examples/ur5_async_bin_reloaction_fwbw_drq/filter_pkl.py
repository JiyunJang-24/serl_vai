import os
import pickle
from typing import List, Dict, Any, Tuple

DONE_REWARD = 1.0   # episode 종료를 판단하는 reward 값
DROP_LEN = 1        # 제거할 episode length (== 1)
OUT_SUFFIX = "_filtered"

def split_into_episodes(data_list: List[Dict[str, Any]], done_reward: float) -> List[List[Dict[str, Any]]]:
    """
    data_list (transition들의 리스트)를 episode들의 리스트로 분할.
    reward == done_reward 를 만나면 episode 종료로 간주.
    """
    episodes: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []

    for t in data_list:
        cur.append(t)
        # 종료 판정
        r = t.get("rewards", None)
        if r == done_reward:
            episodes.append(cur)
            cur = []

    # 마지막이 done 없이 끝났으면(기록이 덜 됐거나) 그것도 하나의 episode로 포함
    if len(cur) > 0:
        episodes.append(cur)

    return episodes

def flatten_episodes(episodes: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """episode 리스트를 다시 transition 리스트로 평탄화."""
    out: List[Dict[str, Any]] = []
    for ep in episodes:
        out.extend(ep)
    return out

def filter_episodes(episodes: List[List[Dict[str, Any]]], drop_len: int) -> Tuple[List[List[Dict[str, Any]]], List[int]]:
    """길이 drop_len 인 episode 제거. (반환: 남은 episodes, 제거된 episode 인덱스들)"""
    kept = []
    dropped_idx = []
    for i, ep in enumerate(episodes):
        if len(ep) == drop_len:
            dropped_idx.append(i)
        else:
            kept.append(ep)
    return kept, dropped_idx

def process_one_pkl(in_path: str, out_path: str) -> Dict[str, Any]:
    with open(in_path, "rb") as f:
        data_list = pickle.load(f)

    if not isinstance(data_list, list):
        raise TypeError(f"{in_path}: pickle root is not a list (got {type(data_list)})")

    episodes = split_into_episodes(data_list, DONE_REWARD)
    orig_lengths = [len(ep) for ep in episodes]

    kept_eps, dropped_idx = filter_episodes(episodes, DROP_LEN)
    kept_lengths = [len(ep) for ep in kept_eps]

    new_data_list = flatten_episodes(kept_eps)

    with open(out_path, "wb") as f:
        pickle.dump(new_data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "in_path": in_path,
        "out_path": out_path,
        "orig_episodes": len(episodes),
        "orig_transitions": len(data_list),
        "orig_lengths": orig_lengths,
        "dropped_episode_indices": dropped_idx,
        "dropped_count": len(dropped_idx),
        "kept_episodes": len(kept_eps),
        "kept_transitions": len(new_data_list),
        "kept_lengths": kept_lengths,
    }

def process_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    out_folder = folder_path.rstrip("/")+ OUT_SUFFIX
    os.makedirs(out_folder, exist_ok=True)

    pkl_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pkl")])
    if not pkl_files:
        print(f"No pickle files found in {folder_path}")
        return

    total_dropped_eps = 0
    total_orig_eps = 0

    for fname in pkl_files:
        in_path = os.path.join(folder_path, fname)
        out_path = os.path.join(out_folder, fname.replace(".pkl", "_noLen1.pkl"))

        info = process_one_pkl(in_path, out_path)

        total_orig_eps += info["orig_episodes"]
        total_dropped_eps += info["dropped_count"]

        print(
            f"[{fname}] "
            f"orig_eps={info['orig_episodes']}, orig_trans={info['orig_transitions']} | "
            f"dropped_len{DROP_LEN}_eps={info['dropped_count']} -> "
            f"kept_eps={info['kept_episodes']}, kept_trans={info['kept_transitions']}"
        )
        if info["dropped_count"] > 0:
            # 어떤 episode들이 제거됐는지 간단히 표시
            # (인덱스는 split된 episode 기준 0-based)
            print(f"  dropped episode indices: {info['dropped_episode_indices']}")

    print("\n====== Summary ======")
    print(f"Input folder:  {folder_path}")
    print(f"Output folder: {out_folder}")
    print(f"Total original episodes: {total_orig_eps}")
    print(f"Total dropped episodes (len=={DROP_LEN}): {total_dropped_eps}")
    print(f"Total kept episodes: {total_orig_eps - total_dropped_eps}")

if __name__ == "__main__":
    folder_path = "/home/vai/Desktop/yujin/serl_vai/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/Pick_up_the_white_plush_from_the_bin_and_place_it_on_the_brown_plate"
    process_folder(folder_path)
