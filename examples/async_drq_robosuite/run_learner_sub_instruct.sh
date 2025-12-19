export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized_sparse_reward_sub_instruction_grasp_critic.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_sparse_robosuite_sub_task_instruct_learner_no_HumanInter_bs_64 \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 64 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --demo_path ./relabel_demos_dense_reward/combined \
    --checkpoint_period 1000 \
    --checkpoint_path /data1/JY/serl/robosuite/checkpoint/sub_task_combined \
    --task_buffer_ratio 75 \
    --divide_demo_buffer True 
