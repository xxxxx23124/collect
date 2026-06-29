from pathlib import Path

import torch

from experiment.DDPM.DataSets.catdataset_64 import CatDataset
from experiment.DDPM.Model.TransUnetVer2 import DiffusionTransUNet
from experiment.DDPM.train import run_training


def main():
    project_dir = Path(__file__).resolve().parent
    sample_dir = project_dir / "Ver2 image"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_size = 64
    dataset = CatDataset(image_size=image_size)
    if len(dataset) == 0:
        print("没有找到猫猫头数据，请先检查 CatDataset 里的路径。")
        return

    run_training(
        model_cls=DiffusionTransUNet,
        dataset=dataset,
        epochs=200,
        batch_size=128,
        accumulation_steps=2,
        lr=2e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        timesteps=1000,
        image_size=image_size,
        sample_dir=sample_dir,
        sample_every=1,
        num_samples=4,
        num_workers=4,
        save_model=False,
        use_monitor=False,
    )


if __name__ == "__main__":
    main()