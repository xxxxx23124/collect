import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def act(self, state, valid_mask):
        """
        Args:
            state: np.ndarray, shape = (2, 15, 15)
            valid_mask: np.ndarray, shape = (225,)

        Returns:
            action: int
            log_prob: float
            value: float
        """
        self.model.eval()

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            valid_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        logits, value = self.model(state_tensor, valid_mask=mask_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.item(),
            value.squeeze(-1).item(),
        )

    def evaluate_actions(self, states, masks, actions):
        """
        PPO 更新时使用。

        Args:
            states: Tensor, shape = (B, 2, 15, 15)
            masks: Tensor, shape = (B, 225)
            actions: Tensor, shape = (B,)

        Returns:
            log_probs: Tensor, shape = (B,)
            entropy: Tensor, shape = (B,)
            values: Tensor, shape = (B,)
        """
        logits, values = self.model(states, valid_mask=masks)

        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = values.squeeze(-1)

        return log_probs, entropy, values
