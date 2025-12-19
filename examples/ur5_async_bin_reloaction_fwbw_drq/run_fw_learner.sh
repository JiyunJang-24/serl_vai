export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized_sparse_reward.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_sparse_reward_fw_learner_seed_1 \
    --seed 1 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --sparse_reward True \
    --grasp_reward True \
    --demo_path ./relabel_demos/fw \
    --checkpoint_period 1000 \
    --checkpoint_path /data1/JY/serl/ur5_relocation/checkpoint/fw_sparse
