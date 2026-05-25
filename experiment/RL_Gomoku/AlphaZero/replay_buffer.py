# alphazero/replay_buffer.py

from collections import deque
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        valid_mask: np.ndarray,
        pi: np.ndarray,
        value: float,
    ):
        """
        Args:
            state:
                shape = (2, board_size, board_size)

            valid_mask:
                shape = (board_size * board_size,)

            pi:
                MCTS target policy, shape = (board_size * board_size,)

            value:
                当前 state 的当前玩家视角最终胜负。
                win = 1, lose = -1, draw = 0
        """
        self.buffer.append(
            (
                state.astype(np.float32),
                valid_mask.astype(np.bool_),
                pi.astype(np.float32),
                np.float32(value),
            )
        )

    def sample(
        self,
        batch_size: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        states = []
        valid_masks = []
        pis = []
        values = []

        for idx in indices:
            state, valid_mask, pi, value = self.buffer[idx]

            states.append(state)
            valid_masks.append(valid_mask)
            pis.append(pi)
            values.append(value)

        states = torch.tensor(
            np.stack(states),
            dtype=torch.float32,
            device=device,
        )

        valid_masks = torch.tensor(
            np.stack(valid_masks),
            dtype=torch.bool,
            device=device,
        )

        pis = torch.tensor(
            np.stack(pis),
            dtype=torch.float32,
            device=device,
        )

        values = torch.tensor(
            np.array(values),
            dtype=torch.float32,
            device=device,
        )

        return states, valid_masks, pis, values
