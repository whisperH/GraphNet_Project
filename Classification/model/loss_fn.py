import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import logging

def loss_fn(predict, target):
    target = one_hot_embedding(target.cpu()).cuda()

    loss = nn.BCEWithLogitsLoss()(predict, target)
    return loss, target


def one_hot_embedding(labels, num_classes=4):
    """Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [B,N].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    B, N = labels.size()
    # labels = labels.view(-1, 1)  # [B,N]->[B*N,1]
    labels = labels.view(int(B * N), 1)
    y = torch.FloatTensor(labels.size()[0], num_classes)  # [B*N, D]
    y.zero_()
    y.scatter_(1, labels, 1)
    return y  # [B*N, D]


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class BCEFocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        #pt = _input
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt+1e-8) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+1e-8)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


# def Dice_loss(inputs, target, beta=1, smooth = 1e-5):
#     # n, c, h, w = inputs.size()
#     # nt, ht, wt, ct = target.size()
#     # if h != ht and w != wt:
#     #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
#     # temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
#     # temp_target = target.view(n, -1, ct)

#     temp_inputs = torch.unsqueeze(inputs.sigmoid(), 2)
#     temp_target = torch.unsqueeze(target, 2)

#     #--------------------------------------------#
#     #   计算dice loss
#     #--------------------------------------------#
#     tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
#     fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
#     fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

#     logging.info("22", tp, fp, fn)
#     score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
#     dice_loss = 1 - torch.mean(score)
#     return dice_loss


# class BinaryDiceLoss(nn.model):
# 	def __init__(self):
# 		super(BinaryDiceLoss, self).__init__()
	
# 	def forward(self, input, targets):
# 		# 获取每个批次的大小 N
# 		N = targets.size()[0]
# 		# 平滑变量
# 		smooth = 1
# 		# 将宽高 reshape 到同一纬度
# 		# input_flat = input.view(N, -1)
# 		# targets_flat = targets.view(N, -1)
	
# 		# 计算交集
# 		intersection = input * targets 
# 		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input.sum(1) + targets.sum(1) + smooth)
# 		# 计算一个批次中平均每张图的损失
# 		loss = 1 - N_dice_eff.sum() / N
# 		return loss


def Dice_loss(input, targets):
        # 获取每个批次的大小 N
		N = targets.size()[0]
		# 平滑变量
		smooth = 1e-5
		# 将宽高 reshape 到同一纬度
		# input_flat = input.view(N, -1)
		# targets_flat = targets.view(N, -1)
	
		# 计算交集
		intersection = input.sigmoid() * targets 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input.sum(1) + targets.sum(1) + smooth)
		# 计算一个批次中平均每张图的损失
		loss = 1 - N_dice_eff.sum() / N
		return loss
