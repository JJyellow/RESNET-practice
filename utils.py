import torch

import torch.optim as optim
import torch.nn as nn

def dice_loss(pred, target):
    smooth = 1e-6
    pred = torch.sigmoid(pred) # 轉換為 0~1 概率
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# 建議結合 BCE 損失以穩定訓練
def hybrid_loss(pred, target):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice