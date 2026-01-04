import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 距离权重函数（当前：反距离加权 IDW）
# ============================================================

def distance_weight(
    sdf_gt: torch.Tensor,
    eps: float = 1.0,
    power: float = 2.0,
    max_weight: float = 10.0,
):
    """
    基于像素 SDF 的距离权重
    w(d) = 1 / (d + eps)^power

    Args:
        sdf_gt: [B, 1, H, W] or [B, H, W]
    """
    d = torch.abs(sdf_gt)
    w = 1.0 / torch.pow(d + eps, power)

    if max_weight is not None:
        w = torch.clamp(w, max=max_weight)

    return w


# ============================================================
# 2. 加权 SDF 数值回归（主 loss）
# ============================================================

class WeightedSDFLoss(nn.Module):
    def __init__(
        self,
        eps=1.0,
        power=2.0,
        max_weight=10.0,
        reduction="mean",
        base_loss="l1",
    ):
        super().__init__()
        self.eps = eps
        self.power = power
        self.max_weight = max_weight
        self.reduction = reduction
        self.base_loss = base_loss

    def forward(self, pred, gt):
        weight = distance_weight(
            gt, eps=self.eps, power=self.power, max_weight=self.max_weight
        )

        if self.base_loss == "l1":
            error = torch.abs(pred - gt)
        elif self.base_loss == "l2":
            error = (pred - gt) ** 2
        else:
            raise ValueError(self.base_loss)

        loss = weight * error

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ============================================================
# 3. Sign Loss（符号一致性，不加权）
# ============================================================

def sign_loss(pred_sdf, gt_sdf):
    """
    防止 inside / outside 翻转
    """
    return torch.mean(F.relu(-pred_sdf * torch.sign(gt_sdf)))


# ============================================================
# 4. Zero-level Loss（零等值线对齐）
# ============================================================

# def zero_level_loss(pred_sdf, gt_sdf, thresh=1):
#     """
#     只在 gt 接近 0 的区域约束 pred
#     thresh: 像素单位
#     """
#     mask = torch.abs(gt_sdf) < thresh
#     if mask.sum() == 0:
#         return torch.tensor(0.0, device=pred_sdf.device)
#     return torch.mean(torch.abs(pred_sdf[mask]))
def zero_level_loss(pred_sdf, gt_sdf, band=3):
    """
    在 GT 符号边界附近 band 像素内，
    约束 pred_sdf 接近 0
    """
    # gt_sdf: [B, 1, H, W]
    sign = torch.sign(gt_sdf)

    B, C, H, W = sign.shape
    boundary = torch.zeros_like(sign, dtype=torch.bool)

    # 上下符号变化
    boundary[:, :, 1:, :] |= (sign[:, :, 1:, :] != sign[:, :, :-1, :])
    # 左右符号变化
    boundary[:, :, :, 1:] |= (sign[:, :, :, 1:] != sign[:, :, :, :-1])

    if boundary.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)

    # ===== band 扩展 =====
    if band > 1:
        kernel = torch.ones((1, 1, band, band), device=pred_sdf.device)
        boundary = F.conv2d(
            boundary.float(), kernel, padding=band // 2
        ) > 0

    # 在 band 邻域内约束 pred_sdf → 0
    loss = torch.abs(pred_sdf)[boundary]

    return loss.mean()

# ============================================================
# 5. 弱 Eikonal Loss（像素 SDF 友好）
# ============================================================

def eikonal_loss(pred_sdf):
    """
    |∇SDF| ≈ 1
    使用离散梯度，弱约束
    """
    dx = pred_sdf[:, :, 1:, :] - pred_sdf[:, :, :-1, :]
    dy = pred_sdf[:, :, :, 1:] - pred_sdf[:, :, :, :-1]

    grad_norm = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2 + 1e-8)

    return torch.mean((grad_norm - 1.0) ** 2)


# ============================================================
# 6. Area Loss（正负区域比例）
# ============================================================

def area_loss(pred_sdf, gt_sdf):
    """
    保持 inside / outside 面积比例
    """
    pred_inside = (pred_sdf < 0).float().mean()
    gt_inside = (gt_sdf < 0).float().mean()
    return torch.abs(pred_inside - gt_inside)


# ============================================================
# 7. 总损失封装
# ============================================================

class TotalSDFLoss(nn.Module):
    def __init__(
        self,
        w_sdf=1.0,
        w_sign=0.1,
        w_zero=0.5,
        w_eik=0.05,
        w_area=0.1,
    ):
        super().__init__()

        self.sdf_loss = WeightedSDFLoss(
            eps=1.0,
            power=2,
            max_weight=10.0,
        )

        self.w_sdf = w_sdf
        self.w_sign = w_sign
        self.w_zero = w_zero
        self.w_eik = w_eik
        self.w_area = w_area

    def forward(self, pred_sdf, gt_sdf):
        loss_sdf = self.sdf_loss(pred_sdf, gt_sdf)
        loss_sign = sign_loss(pred_sdf, gt_sdf)
        loss_zero = zero_level_loss(pred_sdf, gt_sdf)
        loss_eik = eikonal_loss(pred_sdf)
        loss_area = area_loss(pred_sdf, gt_sdf)

        total = (
            self.w_sdf * loss_sdf
            + self.w_sign * loss_sign
            + self.w_zero * loss_zero
            + self.w_eik * loss_eik
            + self.w_area * loss_area
        )

        return total, {
            "sdf": loss_sdf.item(),
            "sign": loss_sign.item(),
            "zero": loss_zero.item(),
            "eik": loss_eik.item(),
            "area": loss_area.item(),
        }
