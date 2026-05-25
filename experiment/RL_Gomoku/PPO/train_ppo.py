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
    )

    model = GomokuTransformer(model_config)

    agent = PPOAgent(
        model=model,
        lr=3e-4,
        weight_decay=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ppo_config = PPOConfig()

    trainer = PPOTrainer(
        env=env,
        agent=agent,
        config=ppo_config,
    )

    trainer.train(iterations=10000)


if __name__ == "__main__":
    main()
