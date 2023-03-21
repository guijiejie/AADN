import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torch.utils.data import DataLoader
import os
import imageio
from attack_utils.RESIDEDataset import RESIDE_Dataset
import argparse
from attack_utils.ssim_loss import SSIMLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def clamp(data, lower_value, upper_value):
    """
    clamping the data into range (lower_limit, upper_limit)
    :param data: tensor
    :param lower_value: tensor
    :param upper_value: tensor
    :return: tensor
    """
    return torch.max(torch.min(data, upper_value), lower_value)


def get_metric(img, ref):
    """
    The functions for evaluation are from skimage:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    :param img: img 1
    :param ref: img 2
    :return: psnr and ssim
    """
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    # img_ = img.copy()
    # ref_ = ref.copy()
    # The functions for evaluation are from scikit-image
    # error_ssim, diff_ssim = structural_similarity(img_, ref_, full=True, multichannel=True)
    error_ssim, diff_ssim = structural_similarity(img, ref, full=True, multichannel=True)
    # specific data_range to 255 if needed
    error_psnr = peak_signal_noise_ratio(img, ref)
    return error_psnr, error_ssim


def tensor2img(ten):
    """
    Convert tensor to numpy
    :param ten: range [0, 1], format (1, C, H, W)
    :return: range [0, 255], (H, W, C)
    """
    ten = ten.clamp(min=0, max=1)
    ten = ten.detach().cpu() * 255.0
    ten = ten.numpy().squeeze(0).transpose(1, 2, 0).astype(np.uint8)
    return ten


def AADN_attack_input(model, hazy, label, epsilon, alpha, attack_iters):
    """
    Attack the model, trying to make the dehazed image close to the hazy input
    :param model: dehazing model
    :param hazy: input image (hazy image)
    :param label: target label
    :param epsilon: epsilon in AADN
    :param alpha: alpha in AADN
    :param attack_iters: iteration times
    :return: attacked dehazed output
    """
    alpha = alpha / 255.0
    epsilon = epsilon / 255.0

    upper_limit, lower_limit = 1, 0

    criterion = nn.MSELoss()

    delta = torch.zeros_like(hazy).to(CUR_DEVICE)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - hazy, upper_limit - hazy)
    delta.requires_grad = True
    for _ in range(attack_iters):
        robust_output = model((hazy + delta))
        # here is -1 * loss
        # loss = -1 * criterion(2 * robust_output - 1, 2 * label - 1)
        loss = -1 * criterion(robust_output, label)
        grad = torch.autograd.grad(loss, [delta])[0].detach()
        d = delta
        g = grad
        x = hazy
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
    max_delta = delta.detach()
    return max_delta


def AADN_attack_predict_or_gt(model, hazy, label, epsilon, alpha, attack_iters, criterion):
    """
    Attack the model, trying to make the dehazed image far from {predict dehazed image}/{ground truth clear image}
    :param model: dehazing model
    :param hazy: input image (hazy image)
    :param label: target label
    :param epsilon: epsilon in AADN
    :param alpha: alpha in AADN
    :param attack_iters: iteration times
    :param criterion: loss function
    :return: attacked dehazed output
    """
    alpha = alpha / 255.0
    epsilon = epsilon / 255.0

    upper_limit, lower_limit = 1, 0

    delta = torch.zeros_like(hazy).to(CUR_DEVICE)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - hazy, upper_limit - hazy)
    delta.requires_grad = True
    for _ in range(attack_iters):
        robust_output = model((hazy + delta))
        loss = criterion(robust_output, label)
        grad = torch.autograd.grad(loss, [delta])[0].detach()
        d = delta
        g = grad
        x = hazy
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
    max_delta = delta.detach()
    return max_delta


def AADN_attack_with_mask(model, hazy, label, epsilon, alpha, attack_iters, hazy_threshold="mean"):
    """
    Attack the model, trying to attack model with mask delta
    :param model: dehazing model
    :param hazy: input image (hazy image)
    :param label: target label, here is predict dehazed image
    :param epsilon: epsilon in AADN
    :param alpha: alpha in AADN
    :param attack_iters: iteration times
    :param hazy_threshold: how to generate mask
    :return: attacked dehazed output
    """
    alpha = alpha / 255.0
    epsilon = epsilon / 255.0

    upper_limit, lower_limit = 1, 0

    criterion = nn.MSELoss()

    delta = torch.zeros_like(hazy).to(CUR_DEVICE)
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - hazy, upper_limit - hazy)
    delta.requires_grad = True

    mask = None
    if hazy_threshold == "mean":
        mean = (hazy - label).mean()
        mask = (hazy - label) > mean
    else:
        raise ValueError("This threshold type not supported!")

    mask = mask.sum(dim=1)
    mask = mask.repeat([1, 3, 1, 1])
    mask = mask > 0
    mask = mask.float()

    for _ in range(attack_iters):
        robust_output = model((hazy + delta))
        loss = criterion(2 * robust_output - 1, 2 * label - 1)
        grad = torch.autograd.grad(loss, [delta])[0].detach()
        d = delta
        g = grad
        x = hazy
        d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta = delta * mask
    max_delta = delta.detach() * mask
    return max_delta.detach()


def attack_with_noise(hazy, epsilon, noise="uniform"):
    """
    attack the model with noise
    :param hazy: hazy image
    :param epsilon: epsilon in AADN
    :param noise: noise type
    :return:
    """
    epsilon = epsilon / 255.0
    delta = torch.zeros(size=hazy.size()).to(CUR_DEVICE)

    if noise == "uniform":
        delta.uniform_(-1 * epsilon, epsilon)

    else:
        raise ValueError("This noise type not supported!")
    delta = clamp(delta, lower_limit - hazy, upper_limit - hazy)
    return delta


def attack_folder(dataloader, network, save_dir,
                  device, params, att_type,
                  save_img, save_npy):
    """

    :param dataloader: dataloader
    :param network: dehazing model
    :param save_dir: path to save the results_attack
    :param device: cpu or cuda
    :param params: parameters for AADN
    :param att_type: predict_mse, ground truth, etc.
    :param save_img: save the image to disk
    :param save_npy: save the npy to disk
    :return:
    """
    img_save_dir = os.path.join(save_dir,
                                "eps" + str(params["epsilon"]) + "_alp" + str(params["alpha"]) + "_iter" + str(
                                    params["attack_iters"]))

    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
        if save_img == "True":
            os.mkdir(os.path.join(img_save_dir, "attacked_hazy_img"))
            os.mkdir(os.path.join(img_save_dir, "attacked_output_img"))

        if save_npy == "True":
            os.mkdir(os.path.join(img_save_dir, "attacked_hazy_npy"))
            os.mkdir(os.path.join(img_save_dir, "attacked_output_npy"))

    # "before" means before attack, "after" denotes after attack
    all_psnr_before = 0
    all_ssim_before = 0
    all_psnr_after = 0
    all_ssim_after = 0

    # metirc(hazy, hazy + delta)
    all_psnr_after_hazy = 0
    all_ssim_after_hazy = 0

    for data in dataloader:
        print("#### ************************************* ####")
        ten_hazy = data["hazy"]
        ten_hazy = ten_hazy.to(device)
        ten_clear = data["gt"]
        ten_clear = ten_clear.to(device)

        # the name of first image, we only attack one image each time
        img_name = data["name"][0]

        # obtain the original predict dehazed image
        # network.eval() is BN is an import part
        with torch.no_grad():
            ten_pred = network(ten_hazy)
        # calculate the metric(predict_image, clear_image)
        psnr, ssim = get_metric(tensor2img(ten_pred), tensor2img(ten_clear))
        print("{}: psnr and ssim before attack: {}, {}".format(img_name, psnr, ssim))
        all_psnr_before += psnr
        all_ssim_before += ssim

        # attack with different type
        if att_type == "input":
            attack_delta = AADN_attack_input(network, ten_hazy, ten_hazy.clone().detach(),
                                             epsilon=params["epsilon"], alpha=params["alpha"],
                                             attack_iters=params["attack_iters"])

        elif att_type == "predict_mse":
            criterion = nn.MSELoss().to(CUR_DEVICE)
            attack_delta = AADN_attack_predict_or_gt(network, ten_hazy, ten_pred.detach().clone(),
                                                     epsilon=params["epsilon"], alpha=params["alpha"],
                                                     attack_iters=params["attack_iters"], criterion=criterion)

        elif att_type == "predict_ssim":
            criterion = SSIMLoss().to(CUR_DEVICE)
            attack_delta = AADN_attack_predict_or_gt(network, ten_hazy, ten_pred.detach().clone(),
                                                     epsilon=params["epsilon"], alpha=params["alpha"],
                                                     attack_iters=params["attack_iters"], criterion=criterion)

        elif att_type == "gt":
            criterion = nn.MSELoss().to(CUR_DEVICE)
            attack_delta = AADN_attack_predict_or_gt(network, ten_hazy, ten_clear.detach(),
                                                     epsilon=params["epsilon"], alpha=params["alpha"],
                                                     attack_iters=params["attack_iters"], criterion=criterion)

        elif att_type == "mask":
            attack_delta = AADN_attack_with_mask(network, ten_hazy, ten_pred.detach(),
                                                 epsilon=params["epsilon"], alpha=params["alpha"],
                                                 attack_iters=params["attack_iters"],
                                                 hazy_threshold="mean")

        elif att_type == "noise":
            attack_delta = attack_with_noise(ten_hazy, epsilon=params["epsilon"])

        else:
            raise ValueError("This attack type is not supported!")

        # obtain the attacked hazy image $I^{\delta}$
        # print("delta max: ", attack_delta.max())
        attacked_hazy = ten_hazy + attack_delta
        # attacked_hazy = attacked_hazy.clamp(min=0, max=1)

        # obtain the attacked predict dehazed image $J_{p}^{\delta}$
        # network.eval() is BN is an import part
        with torch.no_grad():
            attacked_output = network(attacked_hazy)

        # calculate the metircs($J_{p}^{\delta}, J$)
        psnr, ssim = get_metric(tensor2img(attacked_output.clone()), tensor2img(ten_clear.clone()))
        all_psnr_after += psnr
        all_ssim_after += ssim
        print("{}: psnr and ssim after attack: {}, {}".format(img_name, psnr, ssim))

        # calculate the metrics($I^{\dalta} = I + \delta$, I)
        psnr_hazy, ssim_hazy = get_metric(tensor2img(attacked_hazy.clone()), tensor2img(ten_hazy.clone()))
        all_psnr_after_hazy += psnr_hazy
        all_ssim_after_hazy += ssim_hazy
        print("hazy {}: psnr and ssim of hazy after attack: {}, {}".format(img_name, psnr_hazy, ssim_hazy))


        if save_img == "True":
            imageio.imwrite(os.path.join(img_save_dir, "attacked_hazy_img", img_name[:-4] + ".png"),
                            tensor2img(attacked_hazy.clone()))
            imageio.imwrite(os.path.join(img_save_dir, "attacked_output_img", img_name[:-4] + ".png"),
                            tensor2img(attacked_output.clone()))
            # from PIL import Image
            # Image.fromarray(tensor2img(attacked_hazy.clone())).save(os.path.join(img_save_dir, "attacked_hazy_img", img_name[:-4] + ".png"))
            # Image.fromarray(tensor2img(attacked_output.clone())).save(os.path.join(img_save_dir, "attacked_output_img", img_name[:-4] + ".png"))

        if save_npy == "True":
            np.save(os.path.join(img_save_dir, "attacked_hazy_npy", img_name[:-4] + ".npy"),
                    tensor2img(attacked_hazy.clone()))
            np.save(os.path.join(img_save_dir, "attacked_output_npy", img_name[:-4] + ".npy"),
                    tensor2img(attacked_output.clone()))

    # save the metrics to txt file
    txt_name = "eps" + str(params["epsilon"]) + "_alp" + str(params["alpha"]) + "_iter" + str(
        params["attack_iters"]) + ".txt"

    with open(os.path.join(save_dir, txt_name), mode="a") as f:
        info = "before, PSNR: " + str(all_psnr_before / len(dataloader)) + \
               ", SSIM: " + str(all_ssim_before / len(dataloader)) + "\n"
        f.write(info)

        info = "after, PSNR: " + str(all_psnr_after / len(dataloader)) + \
               ", SSIM: " + str(all_ssim_after / len(dataloader)) + "\n"
        f.write(info)

        info = "hazy after, PSNR: " + str(all_psnr_after_hazy / len(dataloader)) \
               + ", SSIM: " + str(all_ssim_after_hazy / len(dataloader)) + "\n"
        f.write(info)
        f.close()


def compute_original(dataloader, network, img_save_dir, device, save_img, save_npy):
    """
    compute the dehazing results_attack before attack operation
    :param dataloader: dataloader
    :param network: dehazing network
    :param img_save_dir: path to save the images
    :param device: cpu or gpu
    :return: None
    """
    if save_img == "True":
        os.mkdir(os.path.join(img_save_dir, "original_hazy_img"))
        os.mkdir(os.path.join(img_save_dir, "original_clear_img"))
        os.mkdir(os.path.join(img_save_dir, "original_output_img"))

    if save_npy == "True":
        os.mkdir(os.path.join(img_save_dir, "original_hazy_npy"))
        os.mkdir(os.path.join(img_save_dir, "original_clear_npy"))
        os.mkdir(os.path.join(img_save_dir, "original_output_npy"))

    all_psnr = 0
    all_ssim = 0

    network.eval()
    with torch.no_grad():
        for data in dataloader:
            print("********************************")
            ten_hazy = data["hazy"]
            ten_hazy = ten_hazy.to(device)
            ten_clear = data["gt"]
            ten_clear = ten_clear.to(device)
            # the name of first image, we only eval one image each time
            img_name = data["name"][0]

            ten_pred = network(ten_hazy)

            # calculate the metrics($J_{p}$, J)
            psnr, ssim = get_metric(tensor2img(ten_pred), tensor2img(ten_clear))
            all_psnr += psnr
            all_ssim += ssim
            print("{}: psnr and ssim before attack: {}, {}".format(img_name, psnr, ssim))

            if save_img == "True":
                imageio.imwrite(os.path.join(img_save_dir, "original_hazy_img", img_name[:-4] + ".png"),
                                tensor2img(ten_hazy))
                imageio.imwrite(os.path.join(img_save_dir, "original_clear_img", img_name[:-4] + ".png"),
                                tensor2img(ten_clear))
                imageio.imwrite(os.path.join(img_save_dir, "original_output_img", img_name[:-4] + ".png"),
                                tensor2img(ten_pred))

            if save_npy == "True":
                np.save(os.path.join(img_save_dir, "original_hazy_npy", img_name[:-4] + ".npy"), tensor2img(ten_hazy))
                np.save(os.path.join(img_save_dir, "original_clear_npy", img_name[:-4] + ".npy"), tensor2img(ten_clear))
                np.save(os.path.join(img_save_dir, "original_output_npy", img_name[:-4] + ".npy"), tensor2img(ten_pred))

        # save the metrics to disk
        with open(os.path.join(img_save_dir, "original_metrics.txt"), mode="a") as f:
            info = "PSNR: " + str(all_psnr / len(dataloader)) + ", SSIM: " + str(all_ssim / len(dataloader)) + "\n"
            f.write(info)
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--net", type=str)
    parser.add_argument("--pth_path", type=str)
    parser.add_argument("--att_type", type=str)
    parser.add_argument("--save_img", type=str, default="False")
    parser.add_argument("--save_npy", type=str, default="False")
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--compute_original", type=str, default="True")

    config = parser.parse_args()

    # fix random seed if we need
    def same_seeds(seed):
        torch.manual_seed(seed)  # （CPU）
        if torch.cuda.is_available():  # （GPU)
            torch.cuda.manual_seed(seed)  # current GPU
        np.random.seed(seed)

    CUR_DEVICE = None
    if config.device == "cpu":
        CUR_DEVICE = torch.device("cpu")
    else:
        CUR_DEVICE = torch.device("cuda")

    upper_limit, lower_limit = 1, 0

    network = None
    results_dir = None
    if config.net == "4KDehazing":
        from methods._4k_dehazing._4k_dehazing import B_transformer
        network = B_transformer()
        results_dir = os.path.join("results_attack/4KDehazing", config.results_dir)

    elif config.net == "GCANet":
        from methods.GCA.GCA import GCANet
        network = GCANet(3, 3, False)
        results_dir = os.path.join("results_attack/GCANet", config.results_dir)

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    network = network.to(CUR_DEVICE)
    # load the network from cpu or gpu checkpoint file
    if config.device == "cpu":
        network.load_state_dict(torch.load(config.pth_path, map_location="cpu"))
    else:
        network.load_state_dict(torch.load(config.pth_path))
    network.eval()

    # chose a dataset
    data_path = None
    if_identity_name = False

    if config.dataset_name == "FoggyCity":
        # data_path = "demo_dataset/FoggyCity/val/"
        data_path = "../dataset/FoggyCity/dehaze/val/"
        if_identity_name = True

    else:
        raise ValueError("Dataset not supported!")

    ds = RESIDE_Dataset(hazy_path=data_path + "hazy/",
                        clear_path=data_path + "clear/",
                        img_size=[config.img_h, config.img_w], if_train=False, if_identity_name=if_identity_name)
    loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)

    if config.compute_original == "True":
        compute_original(dataloader=loader, network=network,
                         img_save_dir=results_dir, device=CUR_DEVICE,
                         save_img=config.save_img, save_npy=config.save_npy)
    # The parameters for attack
    all_epsilon = [0, 2, 4, 6, 8]
    all_alpha = [2]
    all_attack_iters = [10]
    for cur_epsilon in all_epsilon:
        for cur_alpha in all_alpha:
            for cur_attack_iters in all_attack_iters:
                params = {"epsilon": cur_epsilon, "alpha": cur_alpha, "attack_iters": cur_attack_iters}
                attack_folder(dataloader=loader, network=network,
                              save_dir=results_dir, device=CUR_DEVICE, params=params,
                              att_type=config.att_type,
                              save_img=config.save_img, save_npy=config.save_npy)