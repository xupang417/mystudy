import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 1. 基础 SDF 回归（连续场核心）
# ============================================================

def sdf_l1_loss(pred_sdf, gt_sdf):
    """
    Continuous SDF regression.
    """
    return F.l1_loss(pred_sdf, gt_sdf)


# ============================================================
# 2. 边界加权 SDF 回归（连续权重）
# ============================================================

def weighted_sdf_loss(pred_sdf, gt_sdf, sigma=0.1):
    """
    Continuous boundary-aware SDF loss.
    Weight decays smoothly with |gt_sdf|.
    sigma controls effective boundary width (pixel unit).
    """
    weight = torch.exp(-torch.abs(gt_sdf) / sigma)
    return (weight * torch.abs(pred_sdf - gt_sdf)).mean()


# ============================================================
# 3. Zero-level Band Loss（值域带，不用符号）
# ============================================================

def zero_band_loss(pred_sdf, gt_sdf, band=0.001):
    """
    Enforce alignment inside |sdf| < band.
    Continuous band instead of discrete boundary.
    """
    mask1 = torch.abs(gt_sdf) < band
    mask2 = torch.abs(pred_sdf) < band
    mask = mask1 | mask2
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)
    return torch.abs(pred_sdf[mask] - gt_sdf[mask]).mean()


# ============================================================
# 4. 轻量 Eikonal 正则（防止高频震荡）
# ============================================================

def eikonal_loss(pred_sdf):
    """
    Weak eikonal regularization.
    """
    dx = pred_sdf[:, :, 1:, :] - pred_sdf[:, :, :-1, :]
    dy = pred_sdf[:, :, :, 1:] - pred_sdf[:, :, :, :-1]

    dx = dx[:, :, :, :-1]
    dy = dy[:, :, :-1, :]

    grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    return ((grad_mag - 1.0) ** 2).mean()


# ============================================================
# 5. Area Consistency（防止结构消失）
# ============================================================

def area_loss(pred_sdf, gt_sdf):
    """
    Preserve inside / outside area ratio.
    Soft constraint.
    """
    pred_inside = (pred_sdf < 0).float().mean()
    gt_inside = (gt_sdf < 0).float().mean()
    return torch.abs(pred_inside - gt_inside)


# ============================================================
# 6. Total Loss（连续场稳定版）
# ============================================================

class TotalSDFLoss(nn.Module):
    def __init__(
        self,
        w_sdf=1.0,
        w_weighted=3.0,
        w_zero=5.0,
        w_eik=0.1,
        w_area=1.0,
        sigma=5.0,
        band=2.0,
    ):
        super().__init__()
        self.w_sdf = w_sdf
        self.w_weighted = w_weighted
        self.w_zero = w_zero
        self.w_eik = w_eik
        self.w_area = w_area
        self.sigma = sigma
        self.band = band

    def forward(self, pred_sdf, gt_sdf):
        loss_sdf = sdf_l1_loss(pred_sdf, gt_sdf)
        loss_weighted = weighted_sdf_loss(pred_sdf, gt_sdf, self.sigma)
        loss_zero = zero_band_loss(pred_sdf, gt_sdf, self.band)
        loss_eik = eikonal_loss(pred_sdf)
        loss_area = area_loss(pred_sdf, gt_sdf)

        total = (
            self.w_sdf * loss_sdf
            + self.w_weighted * loss_weighted
            + self.w_zero * loss_zero
            + self.w_eik * loss_eik
            + self.w_area * loss_area
        )

        return total, {
            "sdf": loss_sdf.item(),
            "weighted": loss_weighted.item(),
            "zero": loss_zero.item(),
            "eik": loss_eik.item(),
            "area": loss_area.item(),
        }
