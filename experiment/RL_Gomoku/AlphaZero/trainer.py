# alphazero/trainer.py

import torch
import torch.nn.functional as F


class AlphaZeroTrainer:
    def __init__(
        self,
        model,
        optimizer,
        replay_buffer,
        az_config,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.az_config = az_config
        self.device = device

    def train_steps(self, num_steps: int):
        self.model.train()

        logs = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
        }

        if len(self.replay_buffer) < self.az_config.batch_size:
            print(
                f"[Train] ReplayBuffer too small: "
                f"{len(self.replay_buffer)} < {self.az_config.batch_size}"
            )
            return logs

        for step in range(num_steps):
            states, valid_masks, target_pis, target_values = self.replay_buffer.sample(
                batch_size=self.az_config.batch_size,
                device=self.device,
            )

            logits, values = self.model(
                state=states,
                valid_mask=valid_masks,
            )

            values = values.squeeze(-1)

            policy_loss = self._policy_loss(
                logits=logits,
                target_pis=target_pis,
            )

            value_loss = F.mse_loss(
                values,
                target_values,
            )

            loss = (
                self.az_config.policy_loss_weight * policy_loss
                + self.az_config.value_loss_weight * value_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.az_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.az_config.grad_clip_norm,
                )

            self.optimizer.step()

            logs["loss"] += float(loss.item())
            logs["policy_loss"] += float(policy_loss.item())
            logs["value_loss"] += float(value_loss.item())

        for k in logs:
            logs[k] /= max(1, num_steps)

        print(
            f"[Train] "
            f"loss={logs['loss']:.4f}, "
            f"policy_loss={logs['policy_loss']:.4f}, "
            f"value_loss={logs['value_loss']:.4f}"
        )

        return logs

    def _policy_loss(
        self,
        logits: torch.Tensor,
        target_pis: torch.Tensor,
    ) -> torch.Tensor:
        """
        AlphaZero policy loss:

            L_policy = - sum_a pi_a * log p_a
        """
        log_probs = F.log_softmax(logits, dim=-1)

        loss = -(target_pis * log_probs).sum(dim=-1).mean()

        return loss
