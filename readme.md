# The document of code for AADN

The code is for the paper, ''Adversarial Attack and Defense for Dehazing Networks''. 
The arxiv link is, ''https://arxiv.org/pdf/2303.17255.pdf''.

If you have any suggestions, feel free to contact me (Email: cxf_svip@163.com). Thanks.

----------------------------------------------------------

## 1. How to attack 

python AADN_attack.py --dataset_name FoggyCity --net 4KDehazing --pth_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --results_dir FoggyCity_predict_mse_before_defense --att_type predict_mse --save_img True --save_npy False --img_h 256 --img_w 512

you can find more examples in AADN_attack.sh

----------------------------------------------------------

## 2. how to defense

python AADN_4KDehazing_defense.py --results_dir FoggyCity_defense_pred_mse --img_h 256 --img_w 512 --train_batch_size 4 --dataset FoggyCity --loss L2 --total_epoches 40 --ck_path results_train/4KDehazing/FoggyCity/models/4KDehazing_FoggyCity_original.pth --att_type predict_mse

python AADN_GCANet_defense.py --results_dir FoggyCity_defense_pred_mse --img_h 256 --img_w 512 --train_batch_size 4 --dataset FoggyCity --loss L2 --total_epoches 40 --g_lr 0.001 --ck_path results_train/GCANet/FoggyCity/models/GCANet_FoggyCity_original.pth --att_type predict_mse

----------------------------------------------------------

## 3. details about this repo

attack_utils： codes for attack

defense_utils: codes for defense

methods: network define and config

demo_dataset: datasets

results_attack: store the attack results (both before and after defense training)

results_train: store the original training checkpoints and defense training checkpoints

AADN_4KDehazing_defense.py/sh: defense training of 4Kdehazing

AADN_attack.py/sh: attack codes, can be used for 4KDehazing/GCANet

AADN_GCANet_defense.py/sh: defense training of GCANet

----------------------------------------------------------

## 4. Some details
(1) The final dehazing results (displayed in paper) are all calculated by skimage. Therefore, the metrics during training process is just for analyzing the training process (will not be displayed in paper). 
Since all dehazing models set to the same epochs during training process. It is fine to delete the eval code
during process (epochs is fixed).
