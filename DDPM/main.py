import torch

from DDPM.catdataset import CatDataset
from DDPM.train import run_training
from DDPM.UNet import DiffusionUNet_64


if __name__ == "__main__":
    # 1. 实例化数据集
    dataset = CatDataset()

    # 2. 只有当找到图片时才开始训练
    if len(dataset) > 0:
        run_training(
            model_cls=DiffusionUNet_64,
            dataset=dataset,
            epochs=100,  # 训练轮数
            batch_size=32,  # 批次大小 (根据显存调整)
            accumulation_steps=4,  # 梯度累积
            lr=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path="ddpm_cat_model.pth"
        )
    else:
        print("Please fix the dataset paths first.")