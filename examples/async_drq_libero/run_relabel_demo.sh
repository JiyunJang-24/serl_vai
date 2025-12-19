python relabel_demos.py \
    --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_8.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file fw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 0 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_2.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_3.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000

python relabel_demos.py \
    --demo-file bw_pick_the_akita_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demos_5.pkl \
    --sparse_reward 0 \
    --grasp_reward 1 \
    --backward 1 \
    --fw_reward_classifier_ckpt_path classifier_ckpt/fw_checkpoint_1000 \
    --bw_reward_classifier_ckpt_path classifier_ckpt/bw_checkpoint_1000 \
    --grasp_reward_classifier_ckpt_path classifier_ckpt/grasp_checkpoint_1000
    
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
