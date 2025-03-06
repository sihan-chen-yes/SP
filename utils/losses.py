"""
Losses for meshes
Borrowed from: https://github.com/ShichenLiu/SoftRas
Note that I changed the implementation of laplacian matrices from dense tensor to COO sparse tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import cv2 as cv

import network.styleunet.conv2d_gradfix as conv2d_gradfix
import pytorch3d.ops
from pytorch3d.ops.knn import knn_points


class SecondOrderSmoothnessLossForSequence(nn.Module):
    def __init__(self):
        super(SecondOrderSmoothnessLossForSequence, self).__init__()

    def forward(self, x, dim=0):
        assert x.shape[dim] > 3
        a = x.shape[dim]
        a0 = torch.arange(0, a-2).long().to(x.device)
        a1 = torch.arange(1, a-1).long().to(x.device)
        a2 = torch.arange(2, a).long().to(x.device)
        x0 = torch.index_select(x, dim, index=a0)
        x1 = torch.index_select(x, dim, index=a1)
        x2 = torch.index_select(x, dim, index=a2)

        l = (2*x1 - x2 - x0).pow(2)
        return torch.mean(l)


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight):
        return F.mse_loss(pred * weight, target * weight, reduction=self.reduction)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
        if reduction not in ['mean', 'none', 'sum']:
            raise RuntimeError('Unknown reduction type! It should be in ["mean", "none", "sum"]')

    def forward(self, pred, target, weight=None, dim=-1, normalized=True):
        if normalized:      # assumes both ```pred``` and ```target``` have been normalized
            cs = 1 - torch.sum(pred*target, dim=dim)
        else:
            cs = 1 - F.cosine_similarity(pred, target, dim=dim)

        if weight is not None:
            cs = weight * cs
        if self.reduction == 'mean':
            return torch.mean(cs)
        else:
            return torch.sum(cs)


class LeastMagnitudeLoss(nn.Module):
    def __init__(self, average=False):
        super(LeastMagnitudeLoss, self).__init__()
        self.average = average

    def forward(self, x):
        batch_size = x.size(0)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x.sum()


class NegIOULoss(nn.Module):
    def __init__(self, average=False):
        super(NegIOULoss, self).__init__()
        self.average = average

    def forward(self, predict, target):
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims)
        union = (predict + target - predict * target).sum(dims) + 1e-6
        return 1. - (intersect / union).sum() / intersect.nelement()


class KLDLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):
        d = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if self.reduction == 'mean':
            return d / mu.shape[0]
        return d


class PhaseTransitionsPotential(nn.Module):
    """
    Refer to: Phase Transitions, Distance Functions, and Implicit Neural Representations
    """
    def __init__(self, reduction='mean'):
        super(PhaseTransitionsPotential, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        assert torch.all(x >= 0) and torch.all(x <= 1)
        s = 2 * x - 1
        l = s ** 2 - 2 * torch.abs(s) +1
        if self.reduction == 'mean':
            return torch.mean(l)
        return l


class TotalVariationLoss(nn.Module):
    """
    https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
    """
    def __init__(self, scale_factor=None):
        super(TotalVariationLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        assert len(x.shape) == 4
        tv_h = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        l = (tv_h+tv_w) / np.prod(x.shape)
        return l


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def chamfer_loss(P, Q):
    """
    Calculate the Chamfer loss (MSE) between two point sets P and Q.

    Args:
        P (torch.Tensor): A tensor of shape (m, d), representing the first point set.
        Q (torch.Tensor): A tensor of shape (n, d), representing the second point set.

    Returns:
        torch.Tensor: A scalar representing the Chamfer loss.
    """
    P_to_Q_knn_ret = pytorch3d.ops.knn_points(P.unsqueeze(0),
                                              Q.unsqueeze(0))
    p_idx = P_to_Q_knn_ret.idx.squeeze()
    P_to_Q_loss = torch.norm(P - Q[p_idx], dim=-1).mean()

    Q_to_P_knn_ret = pytorch3d.ops.knn_points(Q.unsqueeze(0),
                                              P.unsqueeze(0))
    q_idx = Q_to_P_knn_ret.idx.squeeze()
    Q_to_P_loss = torch.norm(Q - P[q_idx], dim=-1).mean()

    return P_to_Q_loss + Q_to_P_loss

def bound_loss(values, lower_bound=1e-4, upper_bound=5e-2):
    lower_part_mask = values < lower_bound
    upper_part_mask = values > upper_bound
    loss_lower_part = 1 / torch.clamp(values[lower_part_mask], min=1e-7)
    loss_upper_part = torch.square(values[upper_part_mask] - upper_bound)
    loss = torch.zeros_like(values)
    loss[lower_part_mask] = loss_lower_part
    loss[upper_part_mask] = loss_upper_part
    # cv.imwrite('../avatarrex/lbn1/test/lower_part.exr',
    #            lower_part_mask.view(1024, 2048, 3).float().cpu().numpy())
    # cv.imwrite('../avatarrex/lbn1/test/upper_part.exr',
    #            upper_part_mask.view(1024, 2048, 3).float().cpu().numpy())
    return loss.mean()

def depth_map_smooth_loss(depth_map, mask, alpha=10):
    """
    edge-aware Laplacian(second order) smoothness loss
    depth_map: (H, W, 1)
    """

    # grad_x = torch.abs(mask[:, 1:] - mask[:, :-1])  # shape: (H, W-1)
    # grad_y = torch.abs(mask[1:, :] - mask[:-1, :])  # shape: (H-1, W)
    #
    # grad_x_crop = grad_x[1:, :]  # (H-1, W-1)
    # grad_y_crop = grad_y[:, 1:]  # (H-1, W-1)
    #
    # grad = torch.sqrt(torch.clamp(torch.square(grad_x_crop) + torch.square(grad_y_crop), min=1e-6))
    #
    # # grad_disp = (grad.detach().cpu().numpy() * 255).astype(np.uint8)
    # # cv.imshow("Grad Map", grad_disp)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()
    #
    # weight = torch.exp(-alpha * grad) # (H-1, W-1)
    # # weight = torch.exp(-alpha * grad) * mask[:-1, :-1] # (H-1, W-1)
    # weight_crop = weight[1:, 1:]

    # Laplacian computation (second order gradient)
    lap_x = depth_map[:, :-2] - 2 * depth_map[:, 1:-1] + depth_map[:, 2:] # (H, W-2)
    lap_y = depth_map[:-2, :] - 2 * depth_map[1:-1, :] + depth_map[2:, :] # (H-2, W)

    lap_x_crop = lap_x[1:-1, :]  # (H-2, W-2)
    lap_y_crop = lap_y[:, 1:-1]  # (H-2, W-2)
    #TODO
    # loss_x = (weight_crop * torch.abs(lap_x_crop)).mean()
    # loss_y = (weight_crop * torch.abs(lap_y_crop)).mean()

    # loss = weight_crop * torch.abs(lap_x_crop) + weight_crop * torch.abs(lap_y_crop)
    loss = torch.abs(lap_x_crop) + torch.abs(lap_y_crop)
    # foreground mask
    mask_crop = mask[1:-1, 1:-1]
    loss = loss[mask_crop >= 0.5]

    return loss.mean()

def full_aiap_loss(gs_can, gs_obs, n_neighbors=5, mask=None):
    xyz_can = gs_can["positions"]
    xyz_obs = gs_obs["positions"]

    cov_can = gs_can["covariance"]
    cov_obs = gs_obs["covariance"]

    if mask is not None:
        xyz_can = xyz_can[mask]
        xyz_obs = xyz_obs[mask]
        cov_can = cov_can[mask]
        cov_obs = cov_obs[mask]

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0),
                             xyz_can.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    loss_xyz = aiap_loss(xyz_can, xyz_obs, nn_ix=nn_ix)
    loss_cov = aiap_loss(cov_can, cov_obs, nn_ix=nn_ix)

    return loss_xyz, loss_cov

def aiap_loss(x_canonical, x_deformed, n_neighbors=5, nn_ix=None):
    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    if nn_ix is None:
        _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                                 x_canonical.unsqueeze(0),
                                 K=n_neighbors + 1,
                                 return_sorted=True)
        nn_ix = nn_ix.squeeze(0)

    dists_canonical = torch.cdist(x_canonical.unsqueeze(1), x_canonical[nn_ix])[:,0,1:]
    dists_deformed = torch.cdist(x_deformed.unsqueeze(1), x_deformed[nn_ix])[:,0,1:]

    loss = F.l1_loss(dists_canonical, dists_deformed)

    return loss

def binary_cross_entropy(input, target, epsilon=1e-6):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    input = torch.clamp(input, epsilon, 1 - epsilon)
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()