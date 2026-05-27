# alphazero/mcts.py

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class MCTSNode:
    """
    一个节点代表一个棋盘状态。

    注意：
        value_sum / visit_count 存的是当前节点 to_play 玩家视角下的价值。
    """

    def __init__(
        self,
        prior: float,
        to_play: int,
    ):
        self.prior = float(prior)
        self.to_play = to_play

        self.visit_count = 0
        self.value_sum = 0.0

        self.children: Dict[int, MCTSNode] = {}

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroMCTS:
    def __init__(
        self,
        model,
        board_size: int = 15,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = "cuda",
    ):
        self.model = model
        self.board_size = board_size
        self.action_size = board_size * board_size

        self.num_simulations = num_simulations
        self.c_puct = c_puct

        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.device = torch.device(device)

    @torch.no_grad()
    def run(
        self,
        env,
        add_dirichlet_noise: bool = True,
    ) -> Tuple[np.ndarray, MCTSNode]:
        """
        Args:
            env:
                当前环境，不能是终局。

        Returns:
            pi:
                shape = (board_size * board_size,)
                MCTS visit count 归一化后的策略。

            root:
                根节点。
        """
        if env.done:
            raise ValueError("MCTS cannot run on a terminal state.")

        root = MCTSNode(
            prior=1.0,
            to_play=env.current_player,
        )

        self._expand(root, env)

        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            env_copy = env.clone()
            self._simulate(env_copy, root)

        visit_counts = np.zeros(self.action_size, dtype=np.float32)

        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        if visit_counts.sum() > 0:
            pi = visit_counts / visit_counts.sum()
        else:
            valid_actions = env.get_valid_actions()
            pi = np.zeros(self.action_size, dtype=np.float32)
            pi[valid_actions] = 1.0 / len(valid_actions)

        return pi, root

    def _simulate(self, env, root: MCTSNode):
        """
        一次 MCTS 模拟。

        逻辑：
            1. 从 root 按 PUCT 选择到叶子
            2. 如果是终局，得到真实 value
            3. 如果不是终局，用神经网络评估并扩展
            4. 反向传播 value
        """
        node = root
        search_path = [node]

        while node.expanded and not env.done:
            action, node = self._select_child(node)
            env.step(action)
            search_path.append(node)

        if env.done:
            value = self._terminal_value(
                winner=env.winner,
                to_play=node.to_play,
            )
        else:
            value = self._expand(node, env)

        self._backpropagate(search_path, value)

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visit = max(1, node.visit_count)

        for action, child in node.children.items():
            score = self._ucb_score(
                parent=node,
                child=child,
                total_visit=total_visit,
            )

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(
        self,
        parent: MCTSNode,
        child: MCTSNode,
        total_visit: int,
    ) -> float:
        """
        因为 child.value 是 child.to_play 视角，
        而 parent 选择动作时需要 parent.to_play 视角。

        所以 parent 视角下的 Q = -child.value
        """
        q_value = -child.value

        u_value = (
            self.c_puct
            * child.prior
            * math.sqrt(total_visit)
            / (1 + child.visit_count)
        )

        return q_value + u_value

    @torch.no_grad()
    def _expand(self, node: MCTSNode, env) -> float:
        """
        用神经网络评估当前状态，并展开子节点。

        Returns:
            value:
                当前 node.to_play 玩家视角下的 value。
        """
        state_np = env.get_state()
        valid_mask_np = env.get_valid_mask()

        state = torch.from_numpy(state_np).unsqueeze(0).float().to(self.device)
        valid_mask = torch.from_numpy(valid_mask_np).unsqueeze(0).bool().to(self.device)

        logits, value = self.model(
            state=state,
            valid_mask=valid_mask,
        )

        policy = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        value = float(value[0, 0].detach().cpu().item())

        valid_actions = env.get_valid_actions()

        policy_sum = policy[valid_actions].sum()

        if policy_sum <= 0:
            probs = np.ones_like(valid_actions, dtype=np.float32)
            probs /= probs.sum()
        else:
            probs = policy[valid_actions] / policy_sum

        for action, prior in zip(valid_actions, probs):
            child_to_play = 3 - node.to_play
            node.children[int(action)] = MCTSNode(
                prior=float(prior),
                to_play=child_to_play,
            )

        return value

    def _backpropagate(self, search_path, value: float):
        """
        value 是最后一个节点 to_play 视角下的价值。

        往上回传时，每上一层换一个玩家，所以 value 取反。
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

            value = -value

    def _terminal_value(self, winner: int, to_play: int) -> float:
        """
        返回 to_play 玩家视角的终局价值。
        """
        if winner == 0:
            return 0.0

        if winner == to_play:
            return 1.0

        return -1.0

    def _add_dirichlet_noise(self, root: MCTSNode):
        actions = list(root.children.keys())

        if len(actions) == 0:
            return

        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(actions)
        ).astype(np.float32)

        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1.0 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(n)
            )
