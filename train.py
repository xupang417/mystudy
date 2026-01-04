# train.py (改进版：增强记录和评价指标)
import os
import random
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from datetime import datetime

from config import Config
from dataset import SDFInterpolationDataset
from models.UNet import UNet
from models.Loss import TotalSDFLoss
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plot import plot_curves
import json


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_learning_rate(epoch, total_epochs, initial_lr, warmup_epochs, use_cosine=True):
    """学习率调度"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    elif use_cosine:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return initial_lr * cosine_decay
    else:
        return initial_lr


def get_progressive_weights(epoch, max_epochs=200):
    """渐进式损失权重"""
    progress = epoch / max_epochs
    criterion = {
        'w_sdf': 1.0,
        'w_sign': 0.05,
        'w_zero': 0.1,
        'w_eik': 0.05,
        'w_area':0,
    }
    # if progress < 0.2:
    #     criterion.update({
    #         "w_sdf": 1.0,
    #         "w_zero": 0.5,
    #         "w_sign": 0.2,
    #         "w_area": 0.1,
    #         "w_eikonal": 0.0,
    #     })
    # elif progress < 0.7:
    #     criterion.update({
    #         "w_sdf": 1.0,
    #         "w_zero": 0.5,
    #         "w_sign": 0.5,
    #         "w_area": 0.3,
    #         "w_eikonal": 0.1,
    #     })
    # else:
    #     criterion.update({
    #         "w_sdf": 1.0,
    #         "w_zero": 0.5,
    #         "w_sign": 0.5,
    #         "w_area": 0.3,
    #         "w_eikonal": 0.2,
    #     })
    
    return criterion


def compute_evaluation_metrics(pred, gt):
    """
    计算预测精度评价指标
    Args:
        pred: [B, 1, H, W] 预测SDF
        gt: [B, 1, H, W] 真实SDF
    Returns:
        metrics_dict: 包含各种评价指标的字典
    """
    metrics = {}
    
    # 1. 全局SDF准确率指标
    # L1误差
    metrics['l1_error'] = F.l1_loss(pred, gt).item()
    # L2误差 (MSE)
    metrics['l2_error'] = F.mse_loss(pred, gt).item()
    # 相对误差
    abs_gt = torch.abs(gt)
    abs_gt[abs_gt < 1e-6] = 1e-6  # 避免除零
    metrics['relative_error'] = (torch.abs(pred - gt) / abs_gt).mean().item()
    
    # 2. 符号准确率 (内部/外部分类准确率)
    pred_sign = torch.sign(pred)
    gt_sign = torch.sign(gt)
    metrics['sign_accuracy'] = (pred_sign == gt_sign).float().mean().item()
    
    # 3. 等值线准确率指标 (零等值线附近)
    # 零等值线掩码 (边界区域)
    boundary_mask = torch.abs(gt) < 1.0  # 边界附近1像素范围内
    if boundary_mask.sum() > 0:
        # 边界区域L1误差
        metrics['boundary_l1'] = F.l1_loss(pred[boundary_mask], gt[boundary_mask]).item()
        # 边界定位误差
        metrics['boundary_position_error'] = torch.abs(pred[boundary_mask]).mean().item()
    else:
        metrics['boundary_l1'] = 0.0
        metrics['boundary_position_error'] = 0.0
    
    # 4. 距离敏感准确率 (距离边界越近要求越精确)
    with torch.no_grad():
        # 距离权重：距离边界越近权重越大
        distance_weights = 1.0 / (torch.abs(gt) + 1.0)
        weighted_error = (torch.abs(pred - gt) * distance_weights).mean()
        metrics['weighted_l1'] = weighted_error.item()
    
    # 5. 峰值信噪比 (PSNR) - 图像质量评估
    mse = F.mse_loss(pred, gt)
    metrics['psnr'] = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse > 0 else 100.0
    
    return metrics


def save_experiment_config(cfg, save_path):
    """保存实验配置信息"""
    config_dict = {
        'experiment_name': cfg.exp_name,
        'timestamp': datetime.now().isoformat(),
        'config': vars(cfg),
        'environment': {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def main():
    # =============================
    # Config & env
    # =============================
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.exp_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.curve_dir, exist_ok=True)
    
    # 创建指标保存目录
    metrics_dir = os.path.join(cfg.exp_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    logger = setup_logger(cfg.log_dir)
    logger.info(f"Experiment: {cfg.exp_name}")
    logger.info("使用RIFE改进技巧的渐进式训练策略")
    
    # 保存实验配置
    config_save_path = os.path.join(cfg.exp_dir, "experiment_config.json")
    save_experiment_config(cfg, config_save_path)
    logger.info(f"实验配置已保存至: {config_save_path}")
    
    # 记录环境信息
    logger.info(f"设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")

    # =============================
    # Dataset
    # =============================
    split_file = os.path.join(cfg.data_root, "split.json")
    assert os.path.exists(split_file), f"Split file not found: {split_file}"

    with open(split_file, "r") as f:
        split = json.load(f)

    train_cases = split["train"]
    val_cases = split["val"]

    train_ds = SDFInterpolationDataset(
        cfg.data_root,
        case_list=train_cases
    )

    val_ds = SDFInterpolationDataset(
        cfg.data_root,
        case_list=val_cases
    )

    logger.info(
        f"Dataset | train={len(train_ds)}, val={len(val_ds)}"
    )
    # 记录数据集统计信息
    total_cases = len(split.get("train", [])) \
                + len(split.get("val", [])) \
                + len(split.get("test", []))

    dataset_info = {
        "total_cases": total_cases,
        "train_cases": len(split.get("train", [])),
        "val_cases": len(split.get("val", [])),
        "test_cases": len(split.get("test", [])),

        "train_ratio": len(split.get("train", [])) / total_cases if total_cases > 0 else 0,
        "val_ratio": len(split.get("val", [])) / total_cases if total_cases > 0 else 0,
        "test_ratio": len(split.get("test", [])) / total_cases if total_cases > 0 else 0,

        "split_file": split_file,
    }
    with open(os.path.join(metrics_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # =============================
    # Model
    # =============================
    model = UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels
    ).to(device)

    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总数={total_params:,} 可训练={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # =============================
    # Resume
    # =============================
    start_epoch = 1
    best_val = float("inf")

    train_losses = []
    val_losses = []
    detailed_losses = []
    train_metrics_history = []  # 新增：训练指标历史
    val_metrics_history = []    # 新增：验证指标历史

    last_ckpt_path = os.path.join(cfg.ckpt_dir, "last.pth")
    if cfg.resume and os.path.exists(last_ckpt_path):
        ckpt = load_checkpoint(last_ckpt_path, model, optimizer)
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
        detailed_losses = ckpt.get("detailed_losses", [])
        train_metrics_history = ckpt.get("train_metrics_history", [])
        val_metrics_history = ckpt.get("val_metrics_history", [])
        logger.info(f"从epoch {ckpt['epoch']}恢复训练")

    # =============================
    # Early stopping
    # =============================
    patience = cfg.patience
    min_delta = cfg.min_delta
    no_improve_epochs = 0

    logger.info(f"EarlyStopping | patience={patience}, min_delta={min_delta}")

    # =============================
    # Training loop
    # =============================
    for epoch in range(start_epoch, cfg.epochs + 1):
        # RIFE技巧：动态学习率
        current_lr = get_learning_rate(
            epoch, cfg.epochs, cfg.lr, cfg.warmup_epochs, cfg.use_cosine_decay
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # 获取当前阶段的权重
        weights = get_progressive_weights(epoch, cfg.epochs)
        
        # 创建损失函数实例
        total_loss_fn = TotalSDFLoss(
            w_sdf=1.0,
            w_weighted=3.0,
            w_zero=5.0,
            w_eik=0.1,
            w_area=1.0,
            sigma=0.1,
            band=0.001,
        )
        # ---------- Train ----------
        model.train()
        epoch_train_loss = 0.0
        epoch_detailed = {}
        epoch_train_metrics = {}  # 新增：训练指标

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            # 计算损失
            loss, loss_dict = total_loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            
            # 累积详细损失
            for key, value in loss_dict.items():
                if key not in epoch_detailed:
                    epoch_detailed[key] = 0.0
                epoch_detailed[key] += value
            
            # 计算评价指标（每10个batch计算一次以减少开销）
            if batch_idx % 10 == 0:
                batch_metrics = compute_evaluation_metrics(pred, y)
                for key, value in batch_metrics.items():
                    if key not in epoch_train_metrics:
                        epoch_train_metrics[key] = 0.0
                    epoch_train_metrics[key] += value
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sign_acc': f'{batch_metrics.get("sign_accuracy", 0):.3f}',
                'boundary': f'{batch_metrics.get("boundary_l1", 0):.4f}',
                'lr': f'{current_lr:.1e}'
            })

        # 平均损失和指标
        epoch_train_loss /= len(train_loader)
        for key in epoch_detailed:
            epoch_detailed[key] /= len(train_loader)
        for key in epoch_train_metrics:
            epoch_train_metrics[key] /= (len(train_loader) // 10 + 1)
        
        train_losses.append(epoch_train_loss)
        detailed_losses.append(epoch_detailed)
        train_metrics_history.append(epoch_train_metrics)

        # ---------- Validation ----------
        model.eval()
        epoch_val_loss = 0.0
        val_detailed = {}
        epoch_val_metrics = {}  # 新增：验证指标

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss, loss_dict = total_loss_fn(pred, y)
                epoch_val_loss += loss.item()

                # 累积验证详细损失
                for key, value in loss_dict.items():
                    if key not in val_detailed:
                        val_detailed[key] = 0.0
                    val_detailed[key] += value
                
                # 计算验证集评价指标
                batch_val_metrics = compute_evaluation_metrics(pred, y)
                for key, value in batch_val_metrics.items():
                    if key not in epoch_val_metrics:
                        epoch_val_metrics[key] = 0.0
                    epoch_val_metrics[key] += value

        # 平均验证损失和指标
        epoch_val_loss /= len(val_loader)
        for key in val_detailed:
            val_detailed[key] /= len(val_loader)
        for key in epoch_val_metrics:
            epoch_val_metrics[key] /= len(val_loader)
        
        val_losses.append(epoch_val_loss)
        val_metrics_history.append(epoch_val_metrics)

        # ---------- 详细日志 ----------
        logger.info(f"Epoch {epoch:03d} | LR: {current_lr:.2e}")
        logger.info(f"  Train: {epoch_train_loss:.6f} | Val: {epoch_val_loss:.6f}")

        # 记录评价指标
        logger.info("  Train Metrics:")
        logger.info(f"    Sign Accuracy: {epoch_train_metrics.get('sign_accuracy', 0):.4f}")
        logger.info(f"    L1 Error: {epoch_train_metrics.get('l1_error', 0):.4f}")

        
        logger.info("  Val Metrics:")
        logger.info(f"    Sign Accuracy: {epoch_val_metrics.get('sign_accuracy', 0):.4f}")
        logger.info(f"    L1 Error: {epoch_val_metrics.get('l1_error', 0):.4f}")


        # ---------- Save checkpoint ----------
        save_checkpoint(
            last_ckpt_path,
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer,
            best_val=best_val,
            train_losses=train_losses,
            val_losses=val_losses,
            detailed_losses=detailed_losses,
            current_weights=weights,
            train_metrics_history=train_metrics_history,  # 新增
            val_metrics_history=val_metrics_history,      # 新增
        )

        # ---------- Save best ----------
        if epoch_val_loss < best_val - min_delta:
            best_val = epoch_val_loss
            no_improve_epochs = 0
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, "best.pth"),
                epoch=epoch,
                model=model.state_dict(),
                optimizer=optimizer,
                best_val=best_val,
                train_losses=train_losses,
                val_losses=val_losses,
                train_metrics_history=train_metrics_history,  # 新增
                val_metrics_history=val_metrics_history,      # 新增
            )
            logger.info(f"  New best val loss: {best_val:.6f}")
            
            # 保存最佳模型的指标
            best_metrics = {
                'epoch': epoch,
                'best_val_loss': best_val,
                'val_metrics': epoch_val_metrics,
                'train_metrics': epoch_train_metrics
            }
            with open(os.path.join(metrics_dir, "best_metrics.json"), 'w') as f:
                json.dump(best_metrics, f, indent=2)
        else:
            no_improve_epochs += 1

        # ---------- Periodic save ----------
        if epoch % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pth"),
                epoch=epoch,
                model=model.state_dict(),
                best_val=best_val
            )
            
            # 保存当前epoch的详细指标
            epoch_metrics = {
                'epoch': epoch,
                'train_metrics': epoch_train_metrics,
                'val_metrics': epoch_val_metrics,
                'learning_rate': current_lr
            }
            with open(os.path.join(metrics_dir, f"metrics_epoch_{epoch:03d}.json"), 'w') as f:
                json.dump(epoch_metrics, f, indent=2)

        # ---------- Early stopping ----------
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping at epoch {epoch}. Best val = {best_val:.6f}")
            
            # 保存最终结果摘要
            final_summary = {
                'final_epoch': epoch,
                'best_val_loss': best_val,
                'total_epochs_trained': epoch - start_epoch + 1,
                'early_stopping_triggered': True,
                'final_metrics': {
                    'train': epoch_train_metrics,
                    'val': epoch_val_metrics
                }
            }
            with open(os.path.join(metrics_dir, "training_summary.json"), 'w') as f:
                json.dump(final_summary, f, indent=2)
            break

    # =============================
    # 训练完成后的处理
    # =============================
    if no_improve_epochs < patience:
        # 正常完成训练
        final_summary = {
            'final_epoch': cfg.epochs,
            'best_val_loss': best_val,
            'total_epochs_trained': cfg.epochs - start_epoch + 1,
            'early_stopping_triggered': False,
            'final_metrics': {
                'train': train_metrics_history[-1] if train_metrics_history else {},
                'val': val_metrics_history[-1] if val_metrics_history else {}
            }
        }
        with open(os.path.join(metrics_dir, "training_summary.json"), 'w') as f:
            json.dump(final_summary, f, indent=2)

    # =============================
    # 绘制详细损失曲线和指标曲线
    # =============================
    plot_curves(train_losses, val_losses, cfg.curve_dir, detailed_losses, 
                train_metrics_history, val_metrics_history)
    
    logger.info("训练完成")
    logger.info(f"实验数据保存在: {cfg.exp_dir}")
    logger.info(f"最佳验证损失: {best_val:.6f}")


if __name__ == "__main__":
    main()