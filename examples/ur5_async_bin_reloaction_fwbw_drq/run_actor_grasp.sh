export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_grasp_sparse_reward.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_grasp_fw_actor_no_HumanInter \
    --seed 0 \
    --random_steps 200 \
    --sparse_reward True \
    --grasp_reward True \
    --encoder_type resnet-pretrained \
    # --fw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/fw_checkpoint_50000" \
    # --bw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/bw_checkpoint_30000" \
    # --grasp_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/grasp_checkpoint_30000"
