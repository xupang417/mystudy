import os
from datetime import datetime

class Config:
    # ========= Experiment =========
    exp_name = "sdf_unet_v17_Test_newIDWsdf_guiyihua"
    exp_root = "./experiments"

    seed = 42

    # ========= Data =========
    data_root = "data"
    batch_size = 4
    num_workers = 0

    # ========= Model =========
    in_channels = 2
    out_channels = 1
    base_channels = 64

    # ========= Training =========
    epochs = 300
    lr = 3e-4
    weight_decay = 1e-5

    # RIFE风格学习率调度
    warmup_epochs = 10  # 新增：学习率warmup轮数
    use_cosine_decay = True  # 新增：是否使用余弦退火

    # 损失权重
    lambda_sdf = 1.0
    lambda_eikonal = 0.1

    save_every = 10
    resume = True
    resume_ckpt = "best.pth"
    
    # ========= Early Stopping =========
    patience = 40  # 耐心值
    min_delta = 1e-4  

    # ========= Inference =========
    infer = {
        "sdf_a": "./data/group_0159/image1_sdf.npy",
        "sdf_b": "./data/group_0159/image3_sdf.npy",
        "ckpt_name": "best.pth",
        "out_subdir": "infer/case_0159",
        "save_binary": True,
    }

    # ========= Derived Paths =========
    @property
    def exp_dir(self):
        return os.path.join(self.exp_root, self.exp_name)

    @property
    def ckpt_dir(self):
        return os.path.join(self.exp_dir, "checkpoints")

    @property
    def log_dir(self):
        return os.path.join(self.exp_dir, "logs")

    @property
    def curve_dir(self):
        return os.path.join(self.exp_dir, "curves")
    
    @property
    def infer_dir(self):
        return os.path.join(self.exp_dir, self.infer["out_subdir"])