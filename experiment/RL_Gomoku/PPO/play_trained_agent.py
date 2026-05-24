import torch
import numpy as np

from experiment.RL_Gomoku.Env.GomokuEnv import GomokuEnv
from experiment.RL_Gomoku.Model.GomokuTransformer import GomokuTransformer, GomokuConfig


@torch.no_grad()
def select_action(model, state, valid_mask, device, temperature=1.0):
    model.eval()

    state = torch.tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    valid_mask = torch.tensor(
        valid_mask,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    logits, value = model(state, valid_mask=valid_mask)

    logits = logits / temperature

    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()

    return action.item(), value.item()


def print_board(board):
    size = board.shape[0]

    symbols = {
        0: ".",
        1: "X",
        2: "O",
    }

    print("   " + " ".join([f"{i:2d}" for i in range(size)]))
    for r in range(size):
        row = " ".join([f"{symbols[int(board[r, c])]:>2s}" for c in range(size)])
        print(f"{r:2d} {row}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = GomokuEnv(board_size=15)

    config = GomokuConfig()
    model = GomokuTransformer(config).to(device)

    ckpt_path = "./gomoku_ppo_checkpoints/gomoku_ppo_iter_.pt"

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    state, info = env.reset()

    done = False

    while not done:
        action, value = select_action(
            model=model,
            state=state,
            valid_mask=info["valid_mask"],
            device=device,
            temperature=0.8,
        )

        state, reward, done, info = env.step(action)

        print(f"Player {3 - info['current_player'] if not done else info['winner']} action: {action}, pos: {divmod(action, 15)}, value: {value:.3f}")

    print_board(info["board"])
    print("Winner:", info["winner"])


if __name__ == "__main__":
    main()
