import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import ABC, abstractmethod

from experiment.RL_Gomoku.Model.RoPE2D import RoPE2D


class BaseCrossAttention(nn.Module, ABC):
    """
    简化版交叉注意力基类。

    特点：
        1. 不使用 KVCache
        2. 不使用 attention mask
        3. 不使用 causal mask
        4. Query 来自 x
        5. Key/Value 来自 context
        6. 使用位置编码

    输入:
        x:       (B, S_q, D)
        context: (B, S_kv, D_context)

    输出:
        out:     (B, S_q, D)
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

        Cross-Attention 里通常：
            q_proj: 输入维度 d_model
            k_proj: 输入维度 d_context
            v_proj: 输入维度 d_context
        """
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rotary_emb: RoPE2D | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Query 输入，shape = (B, S_q, D)

            context:
                Key/Value 输入，shape = (B, S_kv, D_context)

            rotary_emb:
                RoPE2D 输入. Query, Key = rotary_emb(Query, Key)

        Returns:
            out:
                shape = (B, S_q, D)
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x shape to be (B, S_q, D), got {tuple(x.shape)}"
            )

        if context.ndim != 3:
            raise ValueError(
                f"Expected context shape to be (B, S_kv, D_context), "
                f"got {tuple(context.shape)}"
            )

        B, S_q, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected x dim={self.d_model}, got {D}"
            )

        # 1. 计算 Q/K/V
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # 2. 拆成多头
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # 3. 应用 RoPE2D
        # 注意：RoPE 只作用于 q/k，不作用于 v
        if rotary_emb is not None:
            q, k = rotary_emb(q, k)

        # 4. Cross-Attention
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
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        # 6. 输出投影
        out = self.out_proj(attn_out)
        out = self.proj_dropout(out)

        return out


class CrossAttention(BaseCrossAttention):
    """
    标准交叉注意力。

    Query 来自 x。
    Key/Value 来自 context。
    """

    def _init_projections(self, **kwargs):
        d_model = kwargs["d_model"]
        d_context = kwargs.get("d_context", d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_context, d_model, bias=False)
        self.v_proj = nn.Linear(d_context, d_model, bias=False)

