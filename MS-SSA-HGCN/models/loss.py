# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, predictions, targets):
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return self.ce_loss(predictions, targets.squeeze())


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.ones(2, 1)
        else:
            if torch.is_tensor(alpha):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha)

    def forward(self, predictions, targets):
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        batch_size = predictions.size(0)
        num_classes = predictions.size(1)

        probs = F.softmax(predictions, dim=1)

        class_mask = predictions.data.new(batch_size, num_classes).fill_(0)
        target_ids = targets.view(-1, 1)
        class_mask.scatter_(1, target_ids.data, 1.)

        if predictions.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()

        alpha_weights = self.alpha[target_ids.data.view(-1)]

        class_probs = (probs * class_mask).sum(1).view(-1, 1)
        log_probs = class_probs.log()

        focal_term = torch.pow((1 - class_probs), self.gamma)
        batch_loss = -alpha_weights * focal_term * log_probs

        return batch_loss.sum()
