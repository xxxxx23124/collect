import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding，用于二维网格 token。

    适合输入形状：
        q, k: (B, num_heads, seq_len, head_dim)

    对于 15x15 五子棋：
        height = 15
        width = 15
        seq_len = 225

    设计思路：
        head_dim 被分成两半：
            前一半用于编码 y 方向，也就是 row
            后一半用于编码 x 方向，也就是 col

        所以：
            y_dim = head_dim // 2
            x_dim = head_dim // 2

        每一半内部再做标准 RoPE。
        因此 head_dim 必须能被 4 整除。

    注意：
        输入 token 的 flatten 顺序必须是 row-major：

            (0, 0), (0, 1), (0, 2), ..., (0, W-1),
            (1, 0), (1, 1), (1, 2), ..., (1, W-1),
            ...

        也就是和 tensor.reshape(B, H * W, C) 的默认顺序一致。
    """

    def __init__(
        self,
        head_dim: int,
        height: int,
        width: int,
        base: float = 500.0,
    ):
        super().__init__()

        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim must be divisible by 4, got head_dim={head_dim}"
            )

        self.head_dim = head_dim
        self.height = height
        self.width = width
        self.base = base

        self.axis_dim = head_dim // 2          # 一半给 y，一半给 x
        self.freq_dim = self.axis_dim // 2     # 每个轴内部 RoPE 的频率数量
        self.seq_len = height * width

        # 构建二维 RoPE 缓存
        cos_y, sin_y, cos_x, sin_x = self._build_2d_rope_cache()

        # 形状都是：
        #   (1, 1, H*W, axis_dim)
        #
        # 这样可以自动广播到：
        #   (B, num_heads, H*W, axis_dim)
        self.register_buffer("cos_y", cos_y, persistent=False)
        self.register_buffer("sin_y", sin_y, persistent=False)
        self.register_buffer("cos_x", cos_x, persistent=False)
        self.register_buffer("sin_x", sin_x, persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        """
        构建 RoPE 逆频率。

        返回形状：
            (freq_dim,)
        """
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.freq_dim, dtype=torch.float32) / self.freq_dim)
        )

    def _build_2d_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        构建每个 token 对应的 y/x 坐标。

        对于 H=3, W=4，row-major 顺序为：

            index:  0  1  2  3   4  5  6  7   8  9 10 11
            y:      0  0  0  0   1  1  1  1   2  2  2  2
            x:      0  1  2  3   0  1  2  3   0  1  2  3

        返回：
            y_pos: (H*W,)
            x_pos: (H*W,)
        """
        y = torch.arange(self.height, dtype=torch.float32)
        x = torch.arange(self.width, dtype=torch.float32)

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        y_pos = grid_y.reshape(-1)
        x_pos = grid_x.reshape(-1)

        return y_pos, x_pos

    def _build_axis_cache(
        self,
        positions: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        给某一个轴构建 cos/sin 缓存。

        Args:
            positions:
                shape = (H*W,)
                可以是 y 坐标，也可以是 x 坐标。

            inv_freq:
                shape = (freq_dim,)

        Returns:
            cos:
                shape = (1, 1, H*W, axis_dim)

            sin:
                shape = (1, 1, H*W, axis_dim)

        为什么要 cat 两次？

            标准 rotate_half 写法是：

                x = [x1, x2]
                rotate_half(x) = [-x2, x1]

            所以 cos/sin 需要和 x 的前后两半对齐。

            如果原始频率是：

                freqs = [f1, f2, f3, ...]

            那么扩展后是：

                emb = [f1, f2, f3, ..., f1, f2, f3, ...]

            这样才能和 rotate_half 配合。
        """
        freqs = torch.outer(positions, inv_freq)

        # shape:
        #   (H*W, freq_dim) -> (H*W, axis_dim)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin

    def _build_2d_rope_cache(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建 y 轴和 x 轴的 RoPE cos/sin 缓存。

        Returns:
            cos_y, sin_y, cos_x, sin_x
        """
        inv_freq = self._build_inv_freq()
        y_pos, x_pos = self._build_2d_positions()

        cos_y, sin_y = self._build_axis_cache(y_pos, inv_freq)
        cos_x, sin_x = self._build_axis_cache(x_pos, inv_freq)

        return cos_y, sin_y, cos_x, sin_x

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        RoPE 的核心旋转操作。

        输入：
            x = [x1, x2]

        输出：
            rotate_half(x) = [-x2, x1]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_axis_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        对某一个轴的特征应用 RoPE。

        Args:
            x:
                shape = (B, num_heads, seq_len, axis_dim)

            cos/sin:
                shape = (1, 1, H*W, axis_dim)

        Returns:
            shape = (B, num_heads, seq_len, axis_dim)
        """
        seq_len = x.shape[2]

        cos = cos[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = sin[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)

        return x * cos + self.rotate_half(x) * sin

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        对一个 q 或 k 应用 2D RoPE。

        Args:
            x:
                shape = (B, num_heads, seq_len, head_dim)

        Returns:
            shape = (B, num_heads, seq_len, head_dim)
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected x to have shape (B, num_heads, seq_len, head_dim), "
                f"but got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim}, but got {x.shape[-1]}"
            )

        if x.shape[2] > self.seq_len:
            raise ValueError(
                f"seq_len={x.shape[2]} is larger than height*width={self.seq_len}. "
                f"If you use extra tokens like CLS, apply RoPE only to board tokens."
            )

        # 前一半是 y 方向特征，后一半是 x 方向特征
        y_part, x_part = x.chunk(2, dim=-1)

        y_part = self._apply_axis_rope(y_part, self.cos_y, self.sin_y)
        x_part = self._apply_axis_rope(x_part, self.cos_x, self.sin_x)

        return torch.cat([y_part, x_part], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 q 和 k 应用 2D RoPE。

        Args:
            q:
                shape = (B, num_heads, seq_len, head_dim)

            k:
                shape = (B, num_heads, seq_len, head_dim)

        Returns:
            q_rope, k_rope
        """
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        return q, k
