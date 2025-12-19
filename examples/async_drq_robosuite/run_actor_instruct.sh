export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python async_drq_randomized_sparse_reward_instruction_grasp_critic.py "$@" \
    --actor \
    --render \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_dense_grasp_divide_instruct_actor_no_HumanInter_bs64 \
    --seed 0 \
    --random_steps 200 \
    --encoder_type resnet-pretrained \
    --fw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/classifier_ckpt/fw_checkpoint_1000" \
    --bw_reward_classifier_ckpt_path "/home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/classifier_ckpt/bw_checkpoint_1000" \
    --grasp_divide True
 