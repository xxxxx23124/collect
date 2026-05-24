import torch
import torch.nn as nn


class PPOAgent:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
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
    def act(
        self,
        state,
        valid_mask,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        self.model.eval()

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_tensor = torch.as_tensor(
            valid_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        action, log_prob, value = self.model.act(
            state=state_tensor,
            valid_mask=mask_tensor,
            deterministic=deterministic,
            temperature=temperature,
        )

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    def evaluate_actions(
        self,
        states,
        masks,
        actions,
    ):
        self.model.train()

        dist, values = self.model.get_action_distribution(
            state=states,
            valid_mask=masks,
        )

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values
