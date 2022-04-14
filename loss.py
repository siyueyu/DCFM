from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :]*pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :])-Iand1
            IoU1 = Iand1/(Ior1 + 1e-5)
            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)

        return IoU/b
        #return IoU


class Scale_IoU(nn.Module):
    def __init__(self):
        super(Scale_IoU, self).__init__()
        self.iou = IoU_loss()

    def forward(self, scaled_preds, gt):
        loss = 0
        for pred_lvl in scaled_preds[0:]:
            loss += self.iou(torch.sigmoid(pred_lvl), gt) + self.iou(1-torch.sigmoid(pred_lvl), 1-gt)
        return loss


def compute_cos_dis(x_sup, x_que):
    x_sup = x_sup.view(x_sup.size()[0], x_sup.size()[1], -1)
    x_que = x_que.view(x_que.size()[0], x_que.size()[1], -1)

    x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True)
    x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True)

    x_que_norm = x_que_norm.permute(0, 2, 1)
    x_qs_norm = torch.matmul(x_que_norm, x_sup_norm)

    x_que = x_que.permute(0, 2, 1)

    x_qs = torch.matmul(x_que, x_sup)
    x_qs = x_qs / (x_qs_norm + 1e-5)
    return x_qs


