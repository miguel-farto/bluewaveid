# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:35:51 2021

@author: user01
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)
    return loss

def focal_loss(input, target, OHEM_percent=None, n_classes=2874):
    gamma = 2
    assert target.size() == input.size()

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    invprobs = F.logsigmoid(-input * (target * 2 - 1))
    loss = (invprobs * gamma).exp() * loss

    if OHEM_percent is None:
        return loss.mean()
    else:
        OHEM, _ = loss.topk(k=int(n_classes * OHEM_percent), dim=1, largest=True, sorted=True)
        return OHEM.mean()

def bce_loss(input, target, OHEM_percent=None, n_classes=2874):
    if OHEM_percent is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=True)
        return loss
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        value, index= loss.topk(int(n_classes * OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()

def focal_OHEM(results, labels, labels_onehot, OHEM_percent=100, n_classes=2874):
    batch_size, class_num = results.shape
    labels = labels.view(-1)
    loss0 = bce_loss(results, labels_onehot, OHEM_percent, n_classes)
    loss1 = focal_loss(results, labels_onehot, OHEM_percent, n_classes)
    indexs_ = (labels != class_num).nonzero().view(-1)
    if len(indexs_) == 0:
        return loss0 + loss1
    results_ = results[torch.arange(0,len(results))[indexs_],labels[indexs_]].contiguous()
    loss2 = focal_loss(results_, torch.ones_like(results_).float().cuda()) # add .cuda()
    return loss0 + loss1 + loss2