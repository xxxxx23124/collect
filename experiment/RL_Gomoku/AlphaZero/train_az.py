# alphazero/main.py

import os

import torch

from experiment.RL_Gomoku.Model.GomokuTransformer import GomokuTransformer, GomokuConfig

from experiment.RL_Gomoku.AlphaZero.az_config import AlphaZeroTrainConfig
from experiment.RL_Gomoku.AlphaZero.replay_buffer import ReplayBuffer
from experiment.RL_Gomoku.AlphaZero.self_play import run_self_play_games
from experiment.RL_Gomoku.AlphaZero.trainer import AlphaZeroTrainer


def save_checkpoint(
    model,
    optimizer,
    iteration: int,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(
        save_dir,
        f"az_iter_{iteration}.pt",
    )

    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

    print(f"[Checkpoint] saved to {path}")


def main():
    az_config = AlphaZeroTrainConfig()

    device = torch.device(az_config.device)

    model_config = GomokuConfig(
        board_size=az_config.board_size,
        in_channels=2,
    )

    model = GomokuTransformer(model_config).to(device)

    if az_config.load_dir is not None:
        ckpt = torch.load(az_config.load_dir, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"{az_config.load_dir} loaded.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=az_config.learning_rate,
        weight_decay=az_config.weight_decay,
    )

    replay_buffer = ReplayBuffer(
        capacity=az_config.replay_buffer_size,
    )

    trainer = AlphaZeroTrainer(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        az_config=az_config,
        device=device,
    )

    for iteration in range(1, az_config.num_iterations + 1):
        print("=" * 80)
        print(f"[Iteration] {iteration}/{az_config.num_iterations}")

        # 1. 自我对弈收集数据
        total_samples = run_self_play_games(
            model=model,
            az_config=az_config,
            replay_buffer=replay_buffer,
            device=device,
        )

        print(
            f"[Iteration] self-play samples added: {total_samples}, "
            f"buffer size: {len(replay_buffer)}"
        )

        # 2. 训练网络
        trainer.train_steps(
            num_steps=az_config.train_steps_per_iter,
        )

        # 3. 保存模型
        if iteration % az_config.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iteration,
                save_dir=az_config.save_dir,
            )


if __name__ == "__main__":
    main()
