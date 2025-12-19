export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_randomized_sparse_reward_fwbw_sub_instruct.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_sparse_bw_learner_instruct_no_HumanInter \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw bw \
    --sparse_reward True \
    --grasp_reward True \
    --demo_path ./relabel_demos/bw \
    --checkpoint_period 1000 \
    --checkpoint_path /data1/JY/serl/ur5_relocation/checkpoint/bw_instruct_sparse
