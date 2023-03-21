#!/bin/bash

########## Part 1 ##########
# demo for $L_{P}^{MSE}$
# before defense training:
python AADN_attack.py --dataset_name FoggyCity --net GCANet --pth_path results_train/GCANet/FoggyCity/models/GCANet_FoggyCity_original.pth --results_dir FoggyCity_predict_mse_before_defense --att_type predict_mse --save_img True --save_npy False --img_h 256 --img_w 512
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_predict_mse_before_defense --att_type predict_mse --save_img True --save_npy False --img_h 256 --img_w 512



# after defense training by $L_{def}^{P}$
python AADN_attack.py --dataset_name FoggyCity --net GCANet --pth_path results_train/GCANet/FoggyCity_defense_pred_mse/models/GCANet_FoggyCity_pred_mse.pth --results_dir FoggyCity_predict_mse_after_defense --att_type predict_mse --save_img True --save_npy False --img_h 256 --img_w 512
#python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity_defense_pred_mse/4KDehazing_FoggyCity_pred_mse.pth --results_dir FoggyCity_predict_mse_after_defense --att_type predict_mse --save_img True --save_npy False --img_h 256 --img_w 512



########## Part 2 ##########
# Other attack: L_{G}^{MSE}, $L_{M}^{MSE}$, $L_{I}^{MSE}$, $A_{N} and $L_{P}^{SSIM}$
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_gt_before_defense --att_type gt --save_img True --save_npy False --img_h 256 --img_w 512
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_mask_before_defense --att_type mask --save_img True --save_npy False --img_h 256 --img_w 512
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_input_before_defense --att_type input --save_img True --save_npy False --img_h 256 --img_w 512
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_noise_before_defense --att_type noise --save_img True --save_npy False --img_h 256 --img_w 512
python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_pred_ssim_before_defense --att_type predict_ssim --save_img True --save_npy False --img_h 256 --img_w 512