#!/bin/bash
#BSUB -J GCA_defense
#BSUB -q gpu_v100
#BSUB -o %GCANet.out
#BSUB -e %GCANet.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load anaconda3
python AADN_GCANet_defense.py --results_dir FoggyCity_defense_pred_mse --img_h 256 --img_w 512 --train_batch_size 4 --dataset FoggyCity --loss L2 --total_epoches 40 --g_lr 0.001 --ck_path results_train/GCANet/FoggyCity/models/GCANet_FoggyCity_original.pth --att_type predict_mse