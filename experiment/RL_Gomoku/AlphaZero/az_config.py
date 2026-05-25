# alphazero/az_config.py

from dataclasses import dataclass


@dataclass
class AlphaZeroTrainConfig:
    board_size: int = 15

    # MCTS
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # 自我对弈
    num_self_play_games_per_iter: int = 10
    temperature_moves: int = 20
    max_game_moves: int = 15 * 15

    # Replay Buffer
    replay_buffer_size: int = 100_000

    # 训练
    batch_size: int = 64
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

    # 设备
    device: str = "cuda"