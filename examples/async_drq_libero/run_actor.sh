export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_libero_fwbw_grasp_actor_HumanInter_2000 \
    --seed 0 \
    --random_steps 200 \
    --encoder_type resnet-pretrained \
    --fw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/classifier_ckpt/fw_checkpoint_1000" \
    --bw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/classifier_ckpt/bw_checkpoint_1000" \
    --grasp_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/classifier_ckpt/grasp_checkpoint_1000" \
    --fw_ckpt_path /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/ckpt/fw/fw_checkpoint_5000 \
    --bw_ckpt_path /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/ckpt/bw/bw_checkpoint_3000 
    # --eval_checkpoint_step 31000 \
    # --eval_checkpoint_step 100
    # --demo_path ./relabel_demos/bw \


    # --fw_ckpt_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_fw_096 \
    # --bw_ckpt_path /home/undergrad/code/serl_dev/examples/async_bin_relocation_fwbw_drq/bin_bw_096 \