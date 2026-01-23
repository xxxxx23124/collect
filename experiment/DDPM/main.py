import torch

from experiment.DDPM.DataSets.catdataset_64 import CatDataset
from experiment.DDPM.DataSets.ImageNet_64 import ImagenetDataset
from train import run_training
from experiment.DDPM.Model.TransUNet import DiffusionTransUNet_64


if __name__ == "__main__":
    # 1. 实例化数据集
    dataset = ImagenetDataset()

    # 2. 只有当找到图片时才开始训练
    if len(dataset) > 0:
        run_training(
            model_cls=DiffusionTransUNet_64,
            dataset=dataset,
            epochs=200,  # 训练轮数
            batch_size=28,  # 批次大小 (根据显存调整)
            accumulation_steps=10,  # 梯度累积
            lr=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path="ddpm_cat_model.pth",
            use_monitor=True
        )
    else:
        print("Please fix the dataset paths first.")