# alphazero/augmentation.py

from typing import List, Tuple

import numpy as np


def _transform_2d(
    x: np.ndarray,
    k: int,
    flip: bool,
) -> np.ndarray:
    """
    对二维棋盘做 D4 对称变换。

    Args:
        x:
            shape = (H, W)

        k:
            旋转次数，0/1/2/3，表示逆时针旋转 k * 90 度

        flip:
            是否做水平翻转

    Returns:
        transformed:
            shape = (H, W)
    """
    y = np.rot90(x, k=k, axes=(0, 1))

    if flip:
        y = np.flip(y, axis=1)

    return np.ascontiguousarray(y)


def _transform_state(
    state: np.ndarray,
    k: int,
    flip: bool,
) -> np.ndarray:
    """
    变换 state。

    Args:
        state:
            shape = (C, H, W)

    Returns:
        transformed_state:
            shape = (C, H, W)
    """
    y = np.rot90(state, k=k, axes=(1, 2))

    if flip:
        y = np.flip(y, axis=2)

    return np.ascontiguousarray(y)


def _transform_flat_board(
    x: np.ndarray,
    board_size: int,
    k: int,
    flip: bool,
) -> np.ndarray:
    """
    变换拉平的一维棋盘向量。

    适用于：
        valid_mask: shape = (H * W,)
        pi:         shape = (H * W,)

    Args:
        x:
            shape = (board_size * board_size,)

    Returns:
        transformed:
            shape = (board_size * board_size,)
    """
    x_2d = x.reshape(board_size, board_size)

    y_2d = _transform_2d(
        x=x_2d,
        k=k,
        flip=flip,
    )

    return np.ascontiguousarray(y_2d.reshape(-1))


def augment_sample(
    state: np.ndarray,
    valid_mask: np.ndarray,
    pi: np.ndarray,
    value: float,
    board_size: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    对单个 AlphaZero 样本做 8 倍棋盘对称增强。

    Args:
        state:
            shape = (2, board_size, board_size)

        valid_mask:
            shape = (board_size * board_size,)

        pi:
            shape = (board_size * board_size,)

        value:
            当前玩家视角下的最终胜负

    Returns:
        augmented_samples:
            List of (state, valid_mask, pi, value)
    """
    augmented = []

    for flip in [False, True]:
        for k in range(4):
            aug_state = _transform_state(
                state=state,
                k=k,
                flip=flip,
            )

            aug_valid_mask = _transform_flat_board(
                x=valid_mask,
                board_size=board_size,
                k=k,
                flip=flip,
            )

            aug_pi = _transform_flat_board(
                x=pi,
                board_size=board_size,
                k=k,
                flip=flip,
            )

            augmented.append(
                (
                    aug_state.astype(np.float32),
                    aug_valid_mask.astype(np.bool_),
                    aug_pi.astype(np.float32),
                    float(value),
                )
            )

    return augmented


def augment_samples(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    board_size: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    对一整局 self-play 的样本做 8 倍增强。

    Args:
        samples:
            List of (state, valid_mask, pi, value)

    Returns:
        augmented_samples:
            增强后的样本列表
    """
    augmented = []

    for state, valid_mask, pi, value in samples:
        augmented.extend(
            augment_sample(
                state=state,
                valid_mask=valid_mask,
                pi=pi,
                value=value,
                board_size=board_size,
            )
        )

    return augmented
