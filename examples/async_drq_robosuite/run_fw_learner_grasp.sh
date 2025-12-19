export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized_grasp_reward_grasp_critic.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_robosuite_grasp_fw_learner_no_HumanInter_no_orientation_bs_32 \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 32 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --demo_path ./relabel_demos_dense_reward/fw \
    --checkpoint_period 1000 \
    --checkpoint_path /data1/JY/serl/robosuite/checkpoint/fw
