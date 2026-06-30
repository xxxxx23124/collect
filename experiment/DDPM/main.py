from pathlib import Path

import torch

from experiment.DDPM.DataSets.fractal_dataset import FractalDataset
from experiment.DDPM.Model.TransUnetVer2 import DiffusionTransUNet, TransUNetConfig
from experiment.DDPM.train import run_training


def main():
    project_dir = Path(__file__).resolve().parent
    image_size = 128
    sample_dir = project_dir / f"fractal_{image_size}_samples"
    checkpoint_dir = project_dir / f"fractal_{image_size}_checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = FractalDataset(
        image_size=image_size,
        random_geometric_augment=True,
    )
    if len(dataset) == 0:
        print("没有找到 fractal 数据，请先运行 fractal_generator.py 或检查 FractalDataset 的路径。")
        return

    model_config = TransUNetConfig(
        image_size=image_size,
        rope_max_size=image_size,
        time_emb_dim=192,
        time_hidden_dim=384,
        time_layers=4,
        num_experts=4,
        dense_inc=12,
        stem_channels=48,
        encoder_channels=(48, 96, 192),
        encoder_blocks=(1, 2, 2),
        decoder_channels=(192, 96, 48),
        decoder_blocks=(2, 2, 1),
        bottleneck_channels=384,
        bottleneck_inner_channels=512,
        bottleneck_layers=3,
        bottleneck_heads=8,
        encoder_attn_levels=(2,),
        decoder_attn_levels=(0,)
    )

    run_training(
        model_cls=DiffusionTransUNet,
        model_kwargs={"config": model_config},
        dataset=dataset,
        epochs=50,
        batch_size=28,
        accumulation_steps=5,
        lr=2e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        timesteps=1000,
        image_size=image_size,
        sample_dir=sample_dir,
        checkpoint_dir=checkpoint_dir,
        save_every_steps=1000,
        sample_on_save=True,
        sample_every=0,
        num_samples=4,
        num_workers=4,
        save_model=True,
        use_monitor=False,
    )


if __name__ == "__main__":
    main()