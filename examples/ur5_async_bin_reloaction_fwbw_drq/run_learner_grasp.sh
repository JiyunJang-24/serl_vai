export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.4 && \
python async_drq_grasp_sparse_reward.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd20demos_grasp_fw_learner_no_HumanInter \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw fw \
    --sparse_reward True \
    --grasp_reward True \
    --demo_path ./grasp_demos \
    --checkpoint_period 1000 \
    --checkpoint_path /data1/JY/serl/ur5_relocation/checkpoint/grasp
