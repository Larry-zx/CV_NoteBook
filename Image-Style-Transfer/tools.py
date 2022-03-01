from __future__ import print_function

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms



def image_loader(image_name, imsize):
    # 定义加载器
    loader = transforms.Compose(
        [
            transforms.Resize([imsize, imsize]),  # 将图像变成(imsize,imsize)的图像 使得style和content同大小
            transforms.ToTensor()
        ]
    )
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)  # 将图片的维度从(H,W)变成(1,H,W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return image.to(device, torch.float)


# 生成Gram矩阵
def gram_matrix(input):
    bath, ch, H, W = input.size()  # 这里bath=1

    features = input.view(bath * ch, H * W)  # 进行flatten操作 变成[ch,H*W]大小的矩阵

    G = torch.mm(features, features.t())  # 将矩阵转置后进行内积

    return G.div(bath * ch * H * W)



#数据归一化
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std