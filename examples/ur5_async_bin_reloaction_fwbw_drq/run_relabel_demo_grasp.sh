
python relabel_demos_grasp.py \
    --demo-file fw/fw_pos_2_demos_2025_06_20_20_52_16.pkl \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_50000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000

python relabel_demos_grasp.py \
    --demo-file fw/fw_pos_18_demos_2025_06_20_20_40_25.pkl \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_50000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000


python relabel_demos_grasp.py \
    --demo-file bw/bw_pos_3_demos_2025_06_20_20_52_16.pkl \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_50000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000

python relabel_demos_grasp.py \
    --demo-file bw/bw_pos_17_demos_2025_06_20_20_40_25.pkl \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_50000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000





# python relabel_demos_grasp.py \
#     --demo-file fw/fw_pos__demos_2025_06_11_22_35_38.pkl \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_30000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000


# python relabel_demos_grasp.py \
#     --demo-file fw/fw_pos__demos_2025_06_11_22_40_11.pkl \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_30000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000


# python relabel_demos_grasp.py \
#     --demo-file bw/bw_pos__demos_2025_06_11_22_22_56.pkl \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_30000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000


# python relabel_demos_grasp.py \
#     --demo-file bw/bw_pos__demos_2025_06_11_22_35_38.pkl \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_30000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000

# python relabel_demos_grasp.py \
#     --demo-file bw/bw_pos__demos_2025_06_11_22_40_11.pkl \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_50000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_30000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_30000

# python relabel_demos_grasp.py \
#     --demo-file fw/fw_pos_10_demos_2025_04_03_21_21_20.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000


# python relabel_demos_grasp.py \
#     --demo-file bw/bw_pos_10_demos_2025_04_02_12_53_48.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw/bw_pos_10_demos_2025_04_03_21_21_20.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000



# python relabel_demos_grasp.py \
#     --demo-file fw_pos__10_demos_2025_03_29_14_26_53.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pos__10_demos_2025_03_29_14_26_53.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pos__9_demos_2025_03_29_14_16_24.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pos__1_demos_2025_03_29_13_41_39.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000


# python relabel_demos_grasp.py \
#     --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_3.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_5.pkl \
#     --sparse_reward 0 \
#     --grasp_reward 1 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
#     --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000
    
# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_10000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 0 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_3.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000

# python relabel_demos_grasp.py \
#     --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_5.pkl \
#     --sparse_reward 0 \
#     --backward 1 \
#     --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
#     --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000
