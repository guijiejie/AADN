import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.pre = nn.Conv2d(64, 3, 3, 1, 1)
        self.re = nn.Sigmoid()

    def forward(self, xs):
        x1 = self.inc(xs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.re(self.pre(x))

        return x


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)

        return result


class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(1, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = ConvBlock(16, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.conv3(output)

        return output


class B_transformer(nn.Module):
    def __init__(self):
        super(B_transformer, self).__init__()

        self.guide_r = GuideNN()
        self.guide_g = GuideNN()
        self.guide_b = GuideNN()

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.u_net = UNet(n_channels=3)
        self.u_net_mini = UNet(n_channels=3)
        # self.u_net_mini = UNet_mini(n_channels=3)
        self.smooth = nn.PReLU()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

        self.x_r_fusion = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.downsample = nn.AdaptiveAvgPool2d((64, 256))
        self.p = nn.PReLU()

        self.r_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.g_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.b_point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_u = F.interpolate(x, (320, 320), mode='bicubic', align_corners=True)

        x_r = F.interpolate(x, (256, 256), mode='bicubic', align_corners=True)
        coeff = self.downsample(self.u_net(x_r)).reshape(-1, 12, 16, 16, 16)

        guidance_r = self.guide_r(x[:, 0:1, :, :])
        guidance_g = self.guide_g(x[:, 1:2, :, :])
        guidance_b = self.guide_b(x[:, 2:3, :, :])

        slice_coeffs_r = self.slice(coeff, guidance_r)
        slice_coeffs_g = self.slice(coeff, guidance_g)
        slice_coeffs_b = self.slice(coeff, guidance_b)

        x_u = self.u_net_mini(x_u)
        x_u = F.interpolate(x_u, (x.shape[2], x.shape[3]), mode='bicubic', align_corners=True)

        output_r = self.apply_coeffs(slice_coeffs_r, self.p(self.r_point(x_u)))
        output_g = self.apply_coeffs(slice_coeffs_g, self.p(self.g_point(x_u)))
        output_b = self.apply_coeffs(slice_coeffs_b, self.p(self.b_point(x_u)))

        output = torch.cat((output_r, output_g, output_b), dim=1)
        output = self.fusion(output)
        output = self.p(self.x_r_fusion(output) * x - output + 1)

        return output

# bt = B_transformer()
# data = torch.zeros(1, 3, 256, 256)
# print(bt(data).shape)


if __name__ == "__main__":
    net = B_transformer()
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    print(out.size())