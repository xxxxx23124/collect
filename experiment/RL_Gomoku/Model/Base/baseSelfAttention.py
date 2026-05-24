import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod

from experiment.RL_Gomoku.Model.RoPE2D import RoPE2D


class BaseSelfAttention(nn.Module, ABC):
    """
    五子棋/棋盘 Transformer 使用的简化版自注意力基类。

    特点：
        1. 不使用 KVCache
        2. 不使用 causal mask
        3. 不使用 padding mask
        4. 可选使用 RoPE2D
        5. 输入一次性是完整棋盘 token，例如 15x15 -> 225 tokens

    输入:
        x: (B, S, D)

    输出:
        out: (B, S, D)

    其中:
        B = batch size
        S = token 数量，例如 15x15 棋盘就是 225
        D = d_model
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, "
                f"got d_model={d_model}, num_heads={num_heads}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.attn_dropout = attn_dropout

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        self._init_projections(
            d_model=d_model,
            num_heads=num_heads,
            **kwargs,
        )

        if not all(p is not None for p in [self.q_proj, self.k_proj, self.v_proj]):
            raise RuntimeError(
                "Q, K, V projection layers must be initialized in _init_projections"
            )

        self.out_proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_dropout)

    @abstractmethod
    def _init_projections(self, **kwargs):
        """
        子类实现 Q/K/V 投影层。

        最普通的写法是：

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RoPE2D | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                输入 token，shape = (B, S, D)

            rotary_emb:
                RoPE2D 模块。
                如果传入，则对 q/k 使用二维旋转位置编码。
                如果为 None，则不使用 RoPE。

        Returns:
            out:
                shape = (B, S, D)
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape to be (B, S, D), got {tuple(x.shape)}"
            )

        B, S, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected input dim={self.d_model}, got {D}"
            )

        # 1. 计算 Q/K/V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆成多头
        # (B, S, D) -> (B, H, S, head_dim)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # 3. 应用 RoPE2D
        # 注意：RoPE 只作用于 q/k，不作用于 v
        if rotary_emb is not None:
            q, k = rotary_emb(q, k)

        # 4. 注意力计算
        # 对五子棋棋盘来说：
        #   不需要 causal mask
        #   不需要 padding mask
        #   所有格子 token 都可以互相看见
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )

        # 5. 合并多头
        # (B, H, S, head_dim) -> (B, S, D)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        # 6. 输出投影
        out = self.out_proj(attn_out)
        out = self.proj_dropout(out)

        return out

class SelfAttention(BaseSelfAttention):
    """
    标准多头自注意力。

    Q/K/V 都来自同一个输入 x。
    """

    def _init_projections(self, **kwargs):
        d_model = kwargs["d_model"]

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

