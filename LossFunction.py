# -*- coding:utf-8 -*-
"""
Loss Functions for ArcMDTI (code_new)
=====================================
Provides:
1. FocalLoss (Cross-Entropy formulation for 2-class hyperspherical/linear logits)
2. BCEFocalLoss (Binary Cross-Entropy formulation for one-hot targets)
3. PolyLoss (Polynomial expansion of cross-entropy loss)
4. SupervisedContrastiveLoss / CosineContrastiveLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class / 2-class Focal Loss built on top of Cross-Entropy:
        L = - alpha * (1 - p_t)^gamma * log(p_t)
    Directly compatible with ArcMarginProduct, CosMarginProduct, SphereMarginProduct,
    DirectMarginProduct, and standard nn.Linear([N, 2]) logits.
    """

    def __init__(self, alpha=0.5, gamma=1.0, reduction="mean", label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        targets = targets.long()

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class BCEFocalLoss(nn.Module):
    """
    Binary Cross-Entropy Focal Loss for [N, 2] logits and [N, 1] or [N] targets.
    Retained for backward compatibility.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        if logits.shape[1] == 2 and targets.shape[1] == 1:
            targets_one_hot = torch.zeros_like(logits)
            targets_one_hot.scatter_(1, targets.long(), 1)
            targets = targets_one_hot

        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def contrastive_loss(embed_A, embed_B, labels, margin=0.5):
    """
    Cosine contrastive loss between paired embeddings.
    """
    if labels.dim() > 1:
        labels = labels.view(-1)
    labels = labels.float()
    cosine_sim = F.cosine_similarity(embed_A, embed_B)
    loss = labels * (1.0 - cosine_sim) + (1.0 - labels) * F.relu(cosine_sim - margin)
    return loss.mean()


class PolyLoss(nn.Module):
    """
    Poly-1 Cross-Entropy Loss: L_Poly1 = L_CE + epsilon * (1 - p_t).
    """

    def __init__(self, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if targets.dim() > 1:
            targets = targets.view(-1)
        targets = targets.long()
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        poly_loss = ce_loss + self.epsilon * (1.0 - pt)
        return poly_loss.mean()

