# alphazero/self_play.py

from typing import List, Tuple

import numpy as np

from experiment.RL_Gomoku.Env.GomokuEnv import GomokuEnv
from experiment.RL_Gomoku.AlphaZero.mcts import AlphaZeroMCTS
from experiment.RL_Gomoku.AlphaZero.augmentation import augment_samples

def select_action_from_pi(
    pi: np.ndarray,
    move_index: int,
    temperature_moves: int,
) -> int:
    """
    AlphaZero 常用策略：
        前若干步按 visit 分布采样，增加探索。
        后面直接 argmax。
    """
    if move_index < temperature_moves:
        pi_sum = pi.sum()

        if pi_sum <= 0:
            pi = np.ones_like(pi) / len(pi)
        else:
            pi = pi / pi_sum

        return int(np.random.choice(len(pi), p=pi))

    return int(np.argmax(pi))


def play_self_play_game(
    model,
    az_config,
    device,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    进行一局自我对弈。

    Returns:
        samples:
            每个元素:
                state, valid_mask, pi, z

            z 是该 state 当前玩家视角下的最终结果。
    """
    env = GomokuEnv(
        board_size=az_config.board_size,
        invalid_move_mode="raise",
    )

    env.reset()

    mcts = AlphaZeroMCTS(
        model=model,
        board_size=az_config.board_size,
        num_simulations=az_config.num_simulations,
        c_puct=az_config.c_puct,
        dirichlet_alpha=az_config.dirichlet_alpha,
        dirichlet_epsilon=az_config.dirichlet_epsilon,
        device=device,
    )

    trajectory = []

    move_index = 0

    while not env.done and move_index < az_config.max_game_moves:
        state = env.get_state()
        valid_mask = env.get_valid_mask()
        current_player = env.current_player

        pi, _ = mcts.run(
            env=env,
            add_dirichlet_noise=True,
        )

        action = select_action_from_pi(
            pi=pi,
            move_index=move_index,
            temperature_moves=az_config.temperature_moves,
        )

        trajectory.append(
            {
                "state": state,
                "valid_mask": valid_mask,
                "pi": pi,
                "player": current_player,
            }
        )

        env.step(action)

        move_index += 1

    winner = env.winner

    samples = []

    for item in trajectory:
        player = item["player"]

        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0

        samples.append(
            (
                item["state"],
                item["valid_mask"],
                item["pi"],
                z,
            )
        )

    return samples


def run_self_play_games(
    model,
    az_config,
    replay_buffer,
    device,
):
    model.eval()

    total_samples = 0

    for game_idx in range(az_config.num_self_play_games_per_iter):
        samples = play_self_play_game(
            model=model,
            az_config=az_config,
            device=device,
        )

        if az_config.use_symmetry_augmentation:
            samples_to_add = augment_samples(
                samples=samples,
                board_size=az_config.board_size,
            )
        else:
            samples_to_add = samples

        for state, valid_mask, pi, z in samples_to_add:
            replay_buffer.add(
                state=state,
                valid_mask=valid_mask,
                pi=pi,
                value=z,
            )

        total_samples += len(samples)

        print(
            f"[Self-Play] game={game_idx + 1}/"
            f"{az_config.num_self_play_games_per_iter}, "
            f"moves={len(samples)}"
        )

    return total_samples
