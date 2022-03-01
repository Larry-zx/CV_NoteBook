from __future__ import print_function

import torch

import torchvision.models as models
from torchvision import utils as vutils

from image_style_transfer.config import style_img_path, content_img_path, result_img_path, num_steps, style_weight, \
    content_weight
from image_style_transfer.model import run_style_transfer
from image_style_transfer.tools import image_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128  # 有gpu加速尺寸设置大一点

image_loader(style_img_path, imsize)

style_img = image_loader(style_img_path, imsize)
content_img = image_loader(content_img_path, imsize)

# 卷积模型 采用VGG19
VGG19 = models.vgg19(pretrained=True).features.to(device).eval()

# 选择噪声图片的形式 是要原content图还是白噪声

# input_img = content_img.clone()
input_img = torch.randn(content_img.data.size(), device=device)


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

output = run_style_transfer(VGG19, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img,
                            num_steps, style_weight, content_weight)

vutils.save_image(output,result_img_path)

