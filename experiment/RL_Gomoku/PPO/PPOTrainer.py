import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from experiment.RL_Gomoku.PPO.PPOBuffer import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99

    rollout_episodes: int = 32

    ppo_epochs: int = 4
    batch_size: int = 128

    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    max_grad_norm: float = 1.0

    save_interval: int = 100
    log_interval: int = 1

    save_dir: str = "./gomoku_ppo_checkpoints"


class PPOTrainer:
    def __init__(
        self,
        env,
        agent,
        config: PPOConfig,
    ):
        self.env = env
        self.agent = agent
        self.config = config

        self.buffer = RolloutBuffer()

        os.makedirs(config.save_dir, exist_ok=True)

    def collect_rollouts(self):
        """
        收集若干完整自博弈对局。
        """
        self.buffer.clear()

        episode_rewards = []
        episode_lengths = []
        winners = []

        for _ in range(self.config.rollout_episodes):
            state, info = self.env.reset()

            done = False

            episode_start_idx = len(self.buffer.states)
            episode_players = []

            step_count = 0

            while not done:
                current_player = info["current_player"]
                valid_mask = info["valid_mask"]

                action, log_prob, value = self.agent.act(state, valid_mask)

                self.buffer.add(
                    state=state,
                    mask=valid_mask,
                    action=action,
                    log_prob=log_prob,
                    value=value,
                    player=current_player,
                )

                episode_players.append(current_player)

                next_state, reward, done, info = self.env.step(action)

                state = next_state
                step_count += 1

            winner = info["winner"]
            winners.append(winner)
            episode_lengths.append(step_count)

            # 根据整局结果给这一局的所有动作分配 return
            episode_end_idx = len(self.buffer.states)
            episode_len = episode_end_idx - episode_start_idx

            returns = []

            for t in range(episode_len):
                player = self.buffer.players[episode_start_idx + t]

                if winner == 0:
                    outcome = 0.0
                elif player == winner:
                    outcome = 1.0
                else:
                    outcome = -1.0

                # 离终局越远，折扣越大
                discount_power = episode_len - 1 - t
                ret = outcome * (self.config.gamma ** discount_power)

                returns.append(ret)

            self.buffer.returns.extend(returns)

            if winner == 0:
                episode_rewards.append(0.0)
            else:
                episode_rewards.append(1.0)

        self.buffer.compute_advantages()

        stats = {
            "mean_episode_len": float(np.mean(episode_lengths)),
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "black_win_rate": float(np.mean([1 if w == 1 else 0 for w in winners])),
            "white_win_rate": float(np.mean([1 if w == 2 else 0 for w in winners])),
            "draw_rate": float(np.mean([1 if w == 0 else 0 for w in winners])),
            "num_steps": len(self.buffer.states),
        }

        return stats

    def update(self):
        """
        PPO 更新。
        """
        device = self.agent.device

        states, masks, actions, old_log_probs, returns, advantages = self.buffer.to_tensors(device)

        num_samples = states.size(0)
        indices = np.arange(num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        update_steps = 0

        self.agent.model.train()

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = indices[start:end]

                batch_idx = torch.tensor(
                    batch_idx,
                    dtype=torch.long,
                    device=device,
                )

                b_states = states[batch_idx]
                b_masks = masks[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]

                new_log_probs, entropy, values = self.agent.evaluate_actions(
                    b_states,
                    b_masks,
                    b_actions,
                )

                ratio = torch.exp(new_log_probs - b_old_log_probs)

                unclipped = ratio * b_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * b_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(values, b_returns)

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.agent.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                update_steps += 1

        stats = {
            "loss": total_loss / update_steps,
            "policy_loss": total_policy_loss / update_steps,
            "value_loss": total_value_loss / update_steps,
            "entropy": total_entropy / update_steps,
        }

        return stats

    def save(self, iteration):
        path = os.path.join(
            self.config.save_dir,
            f"gomoku_ppo_iter_{iteration}.pt",
        )

        torch.save(
            {
                "model": self.agent.model.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "iteration": iteration,
            },
            path,
        )

        print(f"[Save] checkpoint saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.agent.device)
        self.agent.model.load_state_dict(ckpt["model"])
        self.agent.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[Load] checkpoint loaded from {path}")

    def train(self, iterations: int):
        for iteration in range(1, iterations + 1):
            start_time = time.time()

            rollout_stats = self.collect_rollouts()
            update_stats = self.update()

            elapsed = time.time() - start_time

            if iteration % self.config.log_interval == 0:
                print(
                    f"Iter {iteration:05d} | "
                    f"steps {rollout_stats['num_steps']:05d} | "
                    f"len {rollout_stats['mean_episode_len']:.1f} | "
                    f"black {rollout_stats['black_win_rate']:.3f} | "
                    f"white {rollout_stats['white_win_rate']:.3f} | "
                    f"draw {rollout_stats['draw_rate']:.3f} | "
                    f"loss {update_stats['loss']:.4f} | "
                    f"pi {update_stats['policy_loss']:.4f} | "
                    f"v {update_stats['value_loss']:.4f} | "
                    f"ent {update_stats['entropy']:.4f} | "
                    f"time {elapsed:.1f}s"
                )

            if iteration % self.config.save_interval == 0:
                self.save(iteration)
