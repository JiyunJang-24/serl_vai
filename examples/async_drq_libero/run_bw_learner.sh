export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaBinRelocation-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_libero_fwbw_grasp_learner_bw_HumanInter_2000 \
    --seed 0 \
    --random_steps 200 \
    --training_starts 200 \
    --critic_actor_ratio 4 \
    --batch_size 32 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --fwbw bw \
    --demo_path ./relabel_demos_dense_reward/bw \
    --checkpoint_period 3000 \
    --checkpoint_path /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_libero/ckpt/bw
