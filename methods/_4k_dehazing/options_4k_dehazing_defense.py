# -*- coding: utf-8 -*-
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--beta1", type=float, default=0.9)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--total_epoches", type=int, default=100)

        self.parser.add_argument("--dataset", type=str)
        self.parser.add_argument("--data_root_train_FoggyCity", type=str, default="../dataset/FoggyCity/dehaze/train/")
        self.parser.add_argument("--data_root_val_FoggyCity", type=str, default="../dataset/FoggyCity/dehaze/val/")

        self.parser.add_argument("--loss", type=str)

        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument("--img_w", type=int)
        self.parser.add_argument("--img_h", type=int)

        self.parser.add_argument("--g_lr", type=float, default=0.001)
        self.parser.add_argument("--train_batch_size", type=int)
        self.parser.add_argument("--val_batch_size", type=int, default=1)
        self.parser.add_argument("--results_dir", type=str)

        self.parser.add_argument("--ck_path", type=str, default=None, help="pretrained pth path")
        self.parser.add_argument("--att_type", type=str)

    def parse(self):
        parser = self.parser.parse_args()
        return parser


if __name__ == "__main__":
    parser = Options()
    parser = parser.parse()
    print(parser)