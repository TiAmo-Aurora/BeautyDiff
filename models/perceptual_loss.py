import torch
import torch.nn as nn
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=21):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:feature_layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, generated, target):
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        loss = nn.functional.mse_loss(gen_features, target_features)
        return loss
