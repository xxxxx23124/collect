# alphazero/az_config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class AlphaZeroTrainConfig:
    board_size: int = 15

    # MCTS
    num_simulations: int = 100
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.05
    dirichlet_epsilon: float = 0.25

    # 自我对弈
    num_self_play_games_per_iter: int = 50
    temperature_moves: int = 15
    max_game_moves: int = 15 * 15

    # Replay Buffer
    replay_buffer_size: int = 500_000
    use_symmetry_augmentation: bool = True

    # 训练
    batch_size: int = 128
    train_steps_per_iter: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    grad_clip_norm: float = 1.0

    # 总迭代次数
    num_iterations: int = 1000

    # 保存
    save_interval: int = 10
    save_dir: str = "gomoku_az_checkpoints"
    load_dir: Optional[str] = "../PPO/gomoku_ppo_checkpoints/gomoku_ppo_iter_200.pt"

    # 设备
    device: str = "cuda"