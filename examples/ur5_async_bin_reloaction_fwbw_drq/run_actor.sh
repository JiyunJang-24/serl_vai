export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized_sparse_reward.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_sparse_reward_fwbw_actor_seed_1 \
    --seed 1 \
    --random_steps 200 \
    --sparse_reward True \
    --grasp_reward True \
    --encoder_type resnet-pretrained \
    --fw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/fw_checkpoint_50000" \
    --bw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/bw_checkpoint_50000" \
    # --grasp_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/grasp_checkpoint_30000"
