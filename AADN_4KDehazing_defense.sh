#!/bin/bash
#BSUB -J 4Kdehazing_defense
#BSUB -q gpu_v100
#BSUB -o %4KDehazing.out
#BSUB -e %4KDehazing.err
#BSUB -gpu "num=1:mode=exclusive_process"
module load anaconda3
python AADN_4KDehazing_defense.py --results_dir FoggyCity_defense_pred_mse --img_h 256 --img_w 512 --train_batch_size 4 --dataset FoggyCity --loss L2 --total_epoches 40 --ck_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --att_type predict_mse