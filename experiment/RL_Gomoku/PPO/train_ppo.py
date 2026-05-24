import torch

from experiment.RL_Gomoku.Env.GomokuEnv import GomokuEnv
from experiment.RL_Gomoku.Model.GomokuTransformer import GomokuTransformer, GomokuConfig

from PPOAgent import PPOAgent
from PPOTrainer import PPOTrainer, PPOConfig


def main():
    env = GomokuEnv(
        board_size=15,
        invalid_move_mode="raise",
    )

    model_config = GomokuConfig(
        board_size=15,
        in_channels=2,
        d_model=256,
        num_heads=8,
        encoder_layers=6,
        decoder_layers=6,
        ffn_up_dim=682,
        attn_dropout=0.0,
        proj_dropout=0.0,
        ffn_dropout=0.0,
    )

    model = GomokuTransformer(model_config)

    agent = PPOAgent(
        model=model,
        lr=3e-4,
        weight_decay=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ppo_config = PPOConfig(
        gamma=0.99,

        rollout_episodes=16,

        ppo_epochs=4,
        batch_size=256,

        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,

        max_grad_norm=1.0,

        save_interval=50,
        log_interval=1,

        save_dir="./gomoku_ppo_checkpoints",
    )

    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=ppo_config,
    )

    trainer.train(iterations=10000)


if __name__ == "__main__":
    main()
