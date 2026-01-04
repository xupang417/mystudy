# datasets/sdf_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SDFInterpolationDataset(Dataset):
    def __init__(self, root_dir, case_list=None):
            self.samples = []

            if case_list is None:
                case_list = sorted(os.listdir(root_dir))

            for case in case_list:
                case_dir = os.path.join(root_dir, case)
                if not os.path.isdir(case_dir):
                    continue

                sdf1 = os.path.join(case_dir, "image1_sdf.npy")
                sdf2 = os.path.join(case_dir, "image2_sdf.npy")
                sdf3 = os.path.join(case_dir, "image3_sdf.npy")

                if all(os.path.exists(p) for p in [sdf1, sdf2, sdf3]):
                    self.samples.append((sdf1, sdf3, sdf2))

            if len(self.samples) == 0:
                raise RuntimeError("No valid SDF triplets found.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sdf_a_path, sdf_b_path, sdf_gt_path = self.samples[idx]

        sdf_a = np.load(sdf_a_path).astype(np.float32)
        sdf_b = np.load(sdf_b_path).astype(np.float32)
        sdf_gt = np.load(sdf_gt_path).astype(np.float32)

        # shape: (H, W) → (1, H, W)
        sdf_a = torch.from_numpy(sdf_a).unsqueeze(0)
        sdf_b = torch.from_numpy(sdf_b).unsqueeze(0)
        sdf_gt = torch.from_numpy(sdf_gt).unsqueeze(0)

        # input: [SDF_A, SDF_B]
        x = torch.cat([sdf_a, sdf_b], dim=0)

        return x, sdf_gt
