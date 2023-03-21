import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import random


def pair_augmentation(img, target, new_size):

    if random.random() > 0.5:
        img = FF.hflip(img)
        target = FF.hflip(target)
    if random.random() > 0.5:
        img = FF.vflip(img)
        target = FF.vflip(target)

    if random.random() > 0.5:
        i, j, h, w = tfs.RandomCrop.get_params(img, output_size=(new_size[0], new_size[1]))
        img = FF.crop(img, i, j, h, w)
        target = FF.crop(target, i, j, h, w)

    return img, target