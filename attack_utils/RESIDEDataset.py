import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as tt
import numpy as np


class RESIDE_Dataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, img_size, if_train, trans_hazy=None, trans_gt=None, if_identity_name=False):
        super(RESIDE_Dataset, self).__init__()

        self.haze_imgs_dir = os.listdir(hazy_path)
        self.haze_imgs = [os.path.join(hazy_path + img) for img in self.haze_imgs_dir]

        self.clear_dir = clear_path

        self.img_size = img_size
        self.if_train = if_train

        # whether the file name of hazy and clear are same
        self.if_identity_name = if_identity_name

        self.trans_hazy = None
        self.trans_gt = None
        if trans_hazy:
            self.trans_hazy = trans_hazy
        else:
            self.trans_hazy = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                     tt.ToTensor()])

        if trans_gt:
            self.trans_gt = trans_gt
        else:
            self.trans_gt = tt.Compose([tt.Resize((self.img_size[0], self.img_size[1])),
                                        tt.ToTensor()])

        self.split = "/"
        # if platform.system() == "Windows":
        #     self.split = "\\"

    def __getitem__(self, index):
        # (1) read the hazy image
        data_hazy = None
        data_gt = None
        if self.haze_imgs[index].endswith(".npy"):
            data_hazy = np.load(self.haze_imgs[index])
            data_hazy = Image.fromarray(data_hazy)
        else:
            data_hazy = Image.open(self.haze_imgs[index]).convert('RGB')

        # (2) read the clear image
        img = self.haze_imgs[index]
        clear_name = None
        if self.if_identity_name:
            clear_name = img.split(self.split)[-1]
        else:
            id = img.split(self.split)[-1].split('_')[0]
            clear_name = id + img[-4:]

        # if the file is stored in '.npy'
        if self.haze_imgs[index].endswith(".npy"):
            data_gt = np.load(os.path.join(self.clear_dir, clear_name))
            data_gt = Image.fromarray(data_gt)
        # stored in image format, like '.jpg'
        else:
            data_gt = Image.open(os.path.join(self.clear_dir, clear_name)).convert('RGB')

        # For testset-ITS-RESIDE. Other datasets do not need this
        data_gt = tt.CenterCrop(data_hazy.size[::-1])(data_gt)

        data_hazy = self.trans_hazy(data_hazy)
        data_gt = self.trans_gt(data_gt)
        tar_data = {"hazy": data_hazy, "gt": data_gt,
                    "name": img.split(self.split)[-1],
                    "hazy_path": self.haze_imgs[index]}

        return tar_data

    def __len__(self):
        return len(self.haze_imgs)