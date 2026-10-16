"""Funciones de pérdida que ignoran los píxeles sin dato (-1)."""
import torch.nn.functional as F
import torch


def masked_bce(logits, target):
    valid = (target != -1).float()
    t = target.clamp(min=0).float()
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), t, reduction='none')
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def masked_dice(logits, target, eps=1.0):
    valid = (target != -1).float()
    p = torch.sigmoid(logits.squeeze(1)) * valid
    t = target.clamp(min=0).float() * valid
    inter = (p * t).sum()
    return 1 - (2 * inter + eps) / (p.sum() + t.sum() + eps)


def masked_bce_dice(logits, target):
    return 0.5 * masked_bce(logits, target) + 0.5 * masked_dice(logits, target)


LOSSES = {'bce': masked_bce, 'dice': masked_dice, 'bce_dice': masked_bce_dice}
