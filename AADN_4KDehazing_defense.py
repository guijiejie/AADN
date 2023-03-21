# -*- coding: utf-8 -*-
import torch.optim as optim
import torch.nn as nn
import torch
from methods._4k_dehazing import _4k_dehazing, options_4k_dehazing_defense
from defense_utils.dataset.RESIDEDataset import RESIDE_Dataset
import os
# from defense_utils import save
# from defense_utils.metric import cal_psnr_ssim
from defense_utils.loss.loss_writer import LossWriter


config = options_4k_dehazing_defense.Options().parse()
device = torch.device("cuda")

data_root_train = None
data_root_val = None
if_identity_name = None
if config.dataset == "FoggyCity":
    data_root_train = config.data_root_train_FoggyCity
    data_root_val = config.data_root_val_FoggyCity
    if_identity_name = True

else:
    raise ValueError("dataset not support")
img_size = [config.img_h, config.img_w]
train_dataset = RESIDE_Dataset(data_root_train, img_size=img_size, if_train=True, if_identity_name=if_identity_name)
val_dataset = RESIDE_Dataset(data_root_val, img_size=img_size, if_train=False, if_identity_name=if_identity_name)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.train_batch_size, shuffle=True,
                                           num_workers=config.num_workers, pin_memory=True,
                                           drop_last=True)

# the val_loader can be deleted in this code if there is no val dataset.
# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size=config.val_batch_size, shuffle=False,
#                                          num_workers=config.num_workers, pin_memory=True,
#                                          drop_last=True)

res_dir = os.path.join("results_train/4KDehazing/", config.results_dir)
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
    os.mkdir(os.path.join(res_dir, "image_temp_test"))
    os.mkdir(os.path.join(res_dir, "models"))
    os.mkdir(os.path.join(res_dir, "ssim_psnr"))
    os.mkdir(os.path.join(res_dir, "loss"))

loss_writer = LossWriter(os.path.join(res_dir, "loss"))
generator = _4k_dehazing.B_transformer().cuda()
optimizer = optim.Adam(generator.parameters(),
                       lr=config.g_lr, betas=(config.beta1, config.beta2))


loss_func = None
if config.loss == "L1":
    loss_func = nn.L1Loss()
elif config.loss == "L2":
    loss_func = nn.MSELoss()

iteration = 0

aadn_criterion = nn.MSELoss()
generator.load_state_dict(torch.load(config.ck_path))
from defense_utils.online_attack import attack_predict_or_gt
teacher = _4k_dehazing.B_transformer().cuda()
teacher.load_state_dict(torch.load(config.ck_path))

for epoch in range(config.total_epoches):
    generator.train()
    for data in train_loader:
        image_haze = data["hazy"].to(device)
        image_clear = data["gt"].to(device)

        # #################################################
        delta = None

        # The epsilon, alpha and attack_iters can be dynamic generated
        if config.att_type == "predict_mse":
            teacher_pred = teacher(image_haze).detach().clone()
            delta = attack_predict_or_gt(model=generator, hazy=image_haze, label=teacher_pred,
                                     epsilon=8, alpha=2, attack_iters=10, criterion=aadn_criterion)

            generated_image_attack = generator(image_haze + delta)
            g_loss_attack = loss_func(generated_image_attack, teacher_pred)

        elif config.att_type == "gt":
            delta = attack_predict_or_gt(model=generator, hazy=image_haze, label=image_clear.clone(),
                                     epsilon=8, alpha=2, attack_iters=10, criterion=aadn_criterion)
            generated_image_attack = generator(image_haze + delta)
            g_loss_attack = loss_func(generated_image_attack, image_clear)

        generated_image_ori = generator(image_haze)
        g_loss_ori = loss_func(generated_image_ori, image_clear)

        g_loss = g_loss_attack + g_loss_ori

        optimizer.zero_grad()
        g_loss.backward()
        optimizer.step()

        loss_writer.add("g_loss", g_loss.item(), iteration)
        iteration += 1
        if iteration % 100 == 0:
            print("Iter {}, Loss is {}".format(iteration, g_loss.item()))

        # #################################################

    if epoch > config.total_epoches - 10:
        torch.save(generator.state_dict(),
                   os.path.join(res_dir, "models", str(epoch) + ".pth"))

    # generator.eval()
    # with torch.no_grad():
    #     num_samples = 0
    #     total_ssim = 0
    #     total_psnr = 0
    #     for data in val_loader:
    #         image_haze = data["hazy"].to(device)
    #         image_clear = data["gt"].to(device)
    #
    #         out = generator(image_haze)
    #         out = torch.clamp(out, min=0, max=1)
    #
    #         num_samples += 1
    #         total_psnr += cal_psnr_ssim.cal_batch_psnr(pred=out,
    #                                                    gt=image_clear)
    #         total_ssim += cal_psnr_ssim.cal_batch_ssim(pred=out,
    #                                                    gt=image_clear)
    #         out_cat = torch.cat((image_haze, out, image_clear), dim=3)
    #         save.save_image(image_tensor=out_cat[0],
    #                         out_name=os.path.join(res_dir, "image_temp_test", data["name"][0][:-4] + ".png"))
    #
    #     psnr = total_psnr / num_samples
    #     ssim = total_ssim / num_samples
    #     with open(os.path.join(res_dir, "ssim_psnr/metric.txt"), mode='a') as f:
    #         info = str(epoch) + " " + str(ssim) + " " + str(psnr) + "\n"
    #         f.write(info)
    #         f.close()
    #
    #     print("4k: ||iterations: {}||, ||PSNR {:.4}||, ||SSIM {:.4}||".format(iteration,
    #                                                           psnr,
    #                                                           ssim))