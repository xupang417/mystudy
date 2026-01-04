# infer.py
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import cv2

from config import Config
from models.UNet import UNet
from train import compute_evaluation_metrics


def load_sdf(path):
    """
    Load sdf .npy and return tensor [1,1,H,W]
    """
    sdf = np.load(path).astype(np.float32)
    return torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0)


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================================================
    # Load split.json
    # ======================================================
    split_file = os.path.join(cfg.data_root, "split.json")
    assert os.path.exists(split_file), f"split.json not found: {split_file}"

    with open(split_file, "r") as f:
        split = json.load(f)

    test_cases = split.get("test", [])
    assert len(test_cases) > 0, "No test cases in split.json"

    # ======================================================
    # Load model
    # ======================================================
    model = UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels
    ).to(device)

    ckpt_name = cfg.infer.get("ckpt_name", "best.pth")
    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ======================================================
    # Output directory
    # ======================================================
    out_root = os.path.join(cfg.exp_dir, "infer", "test")
    os.makedirs(out_root, exist_ok=True)

    all_metrics = []

    # ======================================================
    # Inference
    # ======================================================
    with torch.no_grad():
        for case in tqdm(test_cases, desc="Infer test cases"):
            case_dir = os.path.join(cfg.data_root, case)

            sdf_a_path = os.path.join(case_dir, "image1_sdf.npy")
            sdf_b_path = os.path.join(case_dir, "image3_sdf.npy")
            sdf_gt_path = os.path.join(case_dir, "image2_sdf.npy")

            sdf_a = load_sdf(sdf_a_path).to(device)
            sdf_b = load_sdf(sdf_b_path).to(device)
            sdf_gt = load_sdf(sdf_gt_path).to(device)

            # input: [1,2,H,W]
            x = torch.cat([sdf_a, sdf_b], dim=1)

            pred = model(x)

            # ==================================================
            # Save outputs
            # ==================================================
            case_out_dir = os.path.join(out_root, case)
            os.makedirs(case_out_dir, exist_ok=True)

            pred_sdf = pred.squeeze().cpu().numpy()
            gt_sdf = sdf_gt.squeeze().cpu().numpy()

            np.save(
                os.path.join(case_out_dir, "image2_sdf_pred.npy"),
                pred_sdf
            )

            # --------------------------
            # Binary (pred & gt)
            # --------------------------
            if cfg.infer.get("save_binary", True):
                binary_pred = (pred_sdf < 0).astype(np.uint8) * 255
                binary_gt = (gt_sdf < 0).astype(np.uint8) * 255

                np.save(
                    os.path.join(case_out_dir, "binary_pred.npy"),
                    binary_pred
                )
                np.save(
                    os.path.join(case_out_dir, "binary_gt.npy"),
                    binary_gt
                )

                cv2.imwrite(
                    os.path.join(case_out_dir, "binary_pred.png"),
                    binary_pred
                )
                cv2.imwrite(
                    os.path.join(case_out_dir, "binary_gt.png"),
                    binary_gt
                )

            # ==================================================
            # Metrics
            # ==================================================
            metrics = compute_evaluation_metrics(pred, sdf_gt)
            metrics["case"] = case
            all_metrics.append(metrics)

    # ======================================================
    # Save summary
    # ======================================================
    mean_metrics = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
        if k != "case"
    }

    summary = {
        "num_cases": len(all_metrics),
        "checkpoint": ckpt_path,
        "mean_metrics": mean_metrics,
        "all_metrics": all_metrics
    }

    with open(os.path.join(out_root, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("[Infer] Finished")
    print(f"Cases: {len(all_metrics)}")
    print("Mean metrics:")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"Results saved to: {out_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
