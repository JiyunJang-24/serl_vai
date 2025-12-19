export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
python bc_policy.py "$@" \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_bc_20demos_evaluation \
    --seed 0 \
    --save_model True \
    --batch_size 256 \
    --max_steps 100000 \
    --gripper False \
    --remove_xy False \
    --encoder_type resnet-pretrained \
    --fw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/fw_checkpoint_50000" \
    --bw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/bw_checkpoint_30000" \
    --demo_paths "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/demos/fw" \
    --checkpoint_path "/data1/JY/serl/ur5_relocation/checkpoint/bc_models"

    # --grasp_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/ur5_async_bin_reloaction_fwbw_drq/classifier_ckpt/grasp_checkpoint_30000"
