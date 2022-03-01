
import torch.nn.functional as F
import torch.nn as nn
from image_style_transfer.tools import gram_matrix


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach() #从当前计算图中分离下来的 不再计算其梯度

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) #计算L2距离
        return input


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # 得到Gram矩阵

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)  # 计算L2矩阵
        return input


