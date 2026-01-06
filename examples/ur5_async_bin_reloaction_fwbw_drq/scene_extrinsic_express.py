import pickle
pkl_path = "/home/vai/Desktop/yujin/serl_vai/examples/ur5_async_bin_reloaction_fwbw_drq/vla_demos/Pick_up_the_gray_plush_from_the_bin_and_place_it_on_the_brown_plate/Pick_up_the_gray_plush_from_the_bin_and_place_it_on_the_brown_plate_2025_12_30_03_01_57.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
fr0 = data["frames"][0]
print(fr0.keys())
print(fr0["rMe"].shape, fr0["image"].shape)

if isinstance(data, dict):
    print("keys:", data.keys())
