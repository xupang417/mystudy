import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Sign Loss —— 拓扑约束（最高优先级）
# ============================================================

def sign_loss(pred_sdf, gt_sdf):
    return torch.mean(F.relu(-pred_sdf * torch.sign(gt_sdf)))


# ============================================================
# 2. Band SDF Consistency Loss
# ============================================================

def band_sdf_loss(pred_sdf, gt_sdf, band=3.0):
    mask = torch.abs(gt_sdf) < band
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)

    weight = (band - torch.abs(gt_sdf[mask])) / band
    return torch.mean(weight * torch.abs(pred_sdf[mask] - gt_sdf[mask]))


# ============================================================
# 3. Gradient Alignment Loss
# ============================================================

def gradient_alignment_loss(pred_sdf, gt_sdf, band=3.0):
    mask = torch.abs(gt_sdf) < band
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_sdf.device)

    def gradient(x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dx[:, :, :, :-1], dy[:, :, :-1, :]

    pdx, pdy = gradient(pred_sdf)
    gdx, gdy = gradient(gt_sdf)

    mask = mask[:, :, :-1, :-1]

    dot = pdx * gdx + pdy * gdy
    norm_p = torch.sqrt(pdx**2 + pdy**2 + 1e-6)
    norm_g = torch.sqrt(gdx**2 + gdy**2 + 1e-6)

    cos = dot / (norm_p * norm_g + 1e-6)
    return torch.mean((1.0 - cos)[mask])


# ============================================================
# 4. Area Loss —— 弱兜底
# ============================================================

def area_loss(pred_sdf, gt_sdf):
    return torch.abs(
        (pred_sdf < 0).float().mean()
        - (gt_sdf < 0).float().mean()
    )


# ============================================================
# 5. Weak SDF Regression Loss（新增）
# ============================================================

def weak_sdf_loss(pred_sdf, gt_sdf):
    return torch.mean(torch.abs(pred_sdf - gt_sdf))


# ============================================================
# 6. Total Loss
# ============================================================

class TotalSDFLoss(nn.Module):
    def __init__(
        self,
        w_sign=2.0,
        w_band=1.0,
        w_grad=0.5,
        w_area=0.1,
        w_sdf_weak=0.0,   # ← 默认关闭
        band=3.0,
    ):
        super().__init__()
        self.w_sign = w_sign
        self.w_band = w_band
        self.w_grad = w_grad
        self.w_area = w_area
        self.w_sdf_weak = w_sdf_weak
        self.band = band

    def forward(self, pred_sdf, gt_sdf):
        loss_sign = sign_loss(pred_sdf, gt_sdf)
        loss_band = band_sdf_loss(pred_sdf, gt_sdf, band=self.band)
        loss_grad = gradient_alignment_loss(pred_sdf, gt_sdf, band=self.band)
        loss_area = area_loss(pred_sdf, gt_sdf)
        loss_sdfw = weak_sdf_loss(pred_sdf, gt_sdf)

        total = (
            self.w_sign * loss_sign
            + self.w_band * loss_band
            + self.w_grad * loss_grad
            + self.w_area * loss_area
            + self.w_sdf_weak * loss_sdfw
        )

        return total, {
            "sign": loss_sign.item(),
            "band": loss_band.item(),
            "grad": loss_grad.item(),
            "area": loss_area.item(),
            "sdf_weak": loss_sdfw.item(),
        }
