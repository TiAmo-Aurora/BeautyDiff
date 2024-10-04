import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.proj(emb)


class VGGStyleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        self.slice4 = vgg[12:21]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, makeup_features=None):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        if makeup_features is not None:
            h = h + makeup_features
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)


class MakeupDiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.makeup_extractor = VGGStyleFeatureExtractor()

        self.inc = nn.Conv2d(in_channels, 64, 3, padding=1)

        self.down1 = UNetBlock(64, 128, time_emb_dim)
        self.down2 = UNetBlock(128, 256, time_emb_dim)
        self.down3 = UNetBlock(256, 256, time_emb_dim)

        self.up1 = UNetBlock(256, 128, time_emb_dim, up=True)
        self.up2 = UNetBlock(128, 64, time_emb_dim, up=True)
        self.up3 = UNetBlock(64, 64, time_emb_dim, up=True)

        self.outc = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x, t, makeup):
        t = self.time_mlp(t)
        makeup_features = self.makeup_extractor(makeup)

        x1 = self.inc(x)
        x2 = self.down1(x1, t, makeup_features)
        x3 = self.down2(x2, t, makeup_features)
        x4 = self.down3(x3, t, makeup_features)

        x = self.up1(x4, t, makeup_features)
        x = self.up2(x + x3, t, makeup_features)
        x = self.up3(x + x2, t, makeup_features)
        output = self.outc(x + x1)

        return output


class FaceMakeupDiffusionModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, makeup):
        return self.unet(x, t, makeup)