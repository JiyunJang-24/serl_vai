
python train_reward_classifier.py \
    --positive_demo_paths /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/bc_demos/bw_pos/ \
    --negative_demo_paths /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/bc_demos/bw_neg/ \
    --classifier_ckpt_path /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/classifier_ckpt/ \
    --batch_size 128 \
    --num_epochs 1000


python train_reward_classifier.py \
    --positive_demo_paths /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/bc_demos/fw_pos/ \
    --negative_demo_paths /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/bc_demos/fw_neg/ \
    --classifier_ckpt_path /home/fick17/Desktop/JY/SERL/serl/examples/async_drq_robosuite/classifier_ckpt/ \
    --batch_size 128 \
    --num_epochs 1000
