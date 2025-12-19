
python relabel_demos.py \
    --demo-file fw/fw_pos_10_demos_2025_05_08_01_03_37_dense_rand_init_same_coord.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file fw/fw_pos_10_demos_2025_05_08_01_08_21_dense_rand_init_same_coord.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000


# python relabel_demos.py \
#     --demo-file fw/fw_pos_10_demos_2025_04_29_22_06_40_dense_random_init_90_roll.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file bw/bw_pos_10_demos_2025_05_08_01_03_37_dense_rand_init_same_coord.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file bw/bw_pos_10_demos_2025_05_08_01_08_21_dense_rand_init_same_coord.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000


# python relabel_demos.py \
#     --demo-file bw/bw_pos_10_demos_2025_04_29_22_06_40_dense_random_init_90_roll.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw/bw_pos_10_demos_2025_04_05_20_59_59.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pos__9_demos_2025_03_29_14_16_24.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pos__1_demos_2025_03_29_13_41_39.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000


# python relabel_demos.py \
#     --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_3.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_5.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000
    
# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_10000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos.py \
#     --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_3.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_5.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000
