import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.players = []

    def clear(self):
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.returns.clear()
        self.advantages.clear()
        self.players.clear()

    def add(
        self,
        state,
        mask,
        action,
        log_prob,
        value,
        player,
    ):
        self.states.append(np.array(state, copy=True))
        self.masks.append(np.array(mask, copy=True))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.players.append(player)

    def compute_advantages(self):
        values = np.array(self.values, dtype=np.float32)
        returns = np.array(self.returns, dtype=np.float32)

        advantages = returns - values

        # PPO 常用：优势归一化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        self.advantages = advantages.tolist()

    def to_tensors(self, device):
        states = torch.tensor(
            np.array(self.states),
            dtype=torch.float32,
            device=device,
        )

        masks = torch.tensor(
            np.array(self.masks),
            dtype=torch.bool,
            device=device,
        )

        actions = torch.tensor(
            self.actions,
            dtype=torch.long,
            device=device,
        )

        old_log_probs = torch.tensor(
            self.log_probs,
            dtype=torch.float32,
            device=device,
        )

        returns = torch.tensor(
            self.returns,
            dtype=torch.float32,
            device=device,
        )

        advantages = torch.tensor(
            self.advantages,
            dtype=torch.float32,
            device=device,
        )

        return states, masks, actions, old_log_probs, returns, advantages
