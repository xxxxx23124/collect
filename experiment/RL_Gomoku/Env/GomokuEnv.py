import numpy as np

class GomokuEnv:
    """
    state, reward, done, info = env.step(action)
    判断这个动作是否合法；
    在棋盘上下子；
    判断有没有人赢；
    判断是否平局；
    切换当前玩家；
    返回新的状态、奖励、是否结束等信息。
    """

    def __init__(self, board_size=15, invalid_move_mode="raise"):
        self.board_size = board_size
        self.invalid_move_mode = invalid_move_mode
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.done = False
        self.winner = 0

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None
        self.done = False
        self.winner = 0

        info = self._get_info()
        return self.get_state(), info

    def step(self, action):
        if self.done:
            raise ValueError("Game is already done. Please call reset().")

        if action < 0 or action >= self.board_size * self.board_size:
            raise ValueError(f"Action out of range: {action}")

        row, col = divmod(action, self.board_size)

        if self.board[row, col] != 0:
            if self.invalid_move_mode == "raise":
                raise ValueError(f"Invalid move at {row}, {col}")
            else:
                self.done = True
                self.winner = 3 - self.current_player
                reward = -1.0
                return self.get_state(), reward, self.done, self._get_info()

        # 落子
        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        # 检查胜利
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
            return self.get_state(), reward, self.done, self._get_info()

        # 检查平局
        if not self.get_valid_mask().any():
            self.done = True
            self.winner = 0
            reward = 0.0
            return self.get_state(), reward, self.done, self._get_info()

        # 切换玩家
        self.current_player = 3 - self.current_player

        reward = 0.0
        return self.get_state(), reward, self.done, self._get_info()

    def get_state(self):
        """
        返回当前玩家视角的双通道状态:
        state[0] = 己方棋子
        state[1] = 对方棋子
        shape = (2, board_size, board_size)
        """
        own = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == (3 - self.current_player)).astype(np.float32)
        return np.stack([own, opp], axis=0)

    def get_valid_mask(self):
        return self.board.reshape(-1) == 0

    def get_valid_actions(self):
        return np.flatnonzero(self.board.reshape(-1) == 0)

    def _get_info(self):
        return {
            "valid_mask": self.get_valid_mask(),
            "current_player": self.current_player,
            "last_move": self.last_move,
            "winner": self.winner,
            "board": self.board.copy(),
        }

    def _check_win(self, row, col):
        player = self.board[row, col]
        directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        ]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while (
                0 <= r < self.board_size
                and 0 <= c < self.board_size
                and self.board[r, c] == player
            ):
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

if __name__ == '__main__':
    env = GomokuEnv()
    state, info = env.reset()

    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = np.random.choice(valid_actions)

        state, reward, done, info = env.step(action)

    print("游戏结束")
    print("赢家:", info["winner"])
    print("最后一步:", info["last_move"])
    print(info["board"])