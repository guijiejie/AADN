import torch
from .ssim_function import ssim
from .psnr_function import psnr


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cal_batch_psnr(pred, gt):
    assert pred.size() == gt.size()
    sum_psnr = 0
    for i in range(pred.size(0)):
        cur_psnr = psnr(pred[i], gt[i])
        if cur_psnr >= 100:
            raise ValueError("Maybe an error")
        else:
            sum_psnr += cur_psnr

    ave_psnr = sum_psnr / pred.size(0)
    return ave_psnr


def cal_batch_ssim(pred, gt):
    return ssim(img1=pred, img2=gt).cpu().numpy()
