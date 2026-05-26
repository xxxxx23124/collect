import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from experiment.RL_Gomoku.Model.rmsNorm import RMSNorm
from experiment.RL_Gomoku.Model.RoPE2D import RoPE2D
from experiment.RL_Gomoku.Model.Base.baseSelfAttention import SelfAttention
from experiment.RL_Gomoku.Model.Base.baseCrossAttention import CrossAttention
from experiment.RL_Gomoku.Model.Base.baseSwiGLU import SwiGLU


@dataclass
class GomokuConfig:
    """
    五子棋 Transformer 配置。

    输入:
        state: (B, 2, board_size, board_size)

    输出:
        policy_logits: (B, board_size * board_size)
        value: (B, 1)
    """

    board_size: int = 15
    in_channels: int = 2

    d_model: int = 384
    num_heads: int = 6

    encoder_layers: int = 6
    decoder_layers: int = 6

    # SwiGLU 推荐大约 8/3 * d_model
    ffn_up_dim: Optional[int] = None

    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.0

    # Value head 配置
    # 默认与 d_model 保持一致
    value_hidden_dim: Optional[int] = None
    value_use_tanh: bool = True

    # Policy head 配置
    policy_hidden_dim: Optional[int] = None

    # mask 后的非法动作 logit
    invalid_logit: float = -1e9

    def __post_init__(self):
        if self.board_size <= 0:
            raise ValueError(f"board_size must be positive, got {self.board_size}")

        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {self.in_channels}")

        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, "
                f"got d_model={self.d_model}, num_heads={self.num_heads}"
            )

        if self.encoder_layers < 0:
            raise ValueError(f"encoder_layers must >= 0, got {self.encoder_layers}")

        if self.decoder_layers < 0:
            raise ValueError(f"decoder_layers must >= 0, got {self.decoder_layers}")

        if self.ffn_up_dim is None:
            # LLaMA/SwiGLU 常用近似：8/3 * d_model
            self.ffn_up_dim = int(8 * self.d_model / 3)

        if self.value_hidden_dim is None:
            self.value_hidden_dim = self.d_model

        if self.policy_hidden_dim is None:
            self.policy_hidden_dim = self.d_model

    @property
    def seq_len(self) -> int:
        return self.board_size * self.board_size

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


class BoardCNNTokenizer(nn.Module):
    """
    将棋盘状态编码成 token 序列。

    输入:
        x: (B, 2, H, W)

    输出:
        tokens: (B, H * W, d_model)
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        hidden_dim = config.d_model // 2

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=config.d_model,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, config.d_model),
            nn.SiLU(),
        )

        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)

        # (B, C, H, W) -> (B, H * W, C)
        x = x.flatten(2).transpose(1, 2).contiguous()

        x = self.final_norm(x)

        return x


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block。

    Pre-Norm:
        x = x + SelfAttention(Norm(x))
        x = x + FFN(Norm(x))
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
        )

        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            input_dim=config.d_model,
            output_dim=config.d_model,
            up_proj_dim=config.ffn_up_dim,
            hidden_dropout=config.ffn_dropout,
            output_dropout=config.ffn_dropout,
        )

    def forward(self, x: torch.Tensor, rope: RoPE2D) -> torch.Tensor:
        x = x + self.self_attn(
            self.attn_norm(x),
            rotary_emb=rope,
        )

        x = x + self.ffn(
            self.ffn_norm(x),
        )

        return x


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block。

    用于 policy head：
        1. decoder token 自注意力
        2. decoder token 对 encoder 全局特征做 cross attention
        3. FFN
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        self.self_attn_norm = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
        )

        self.cross_attn_norm = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_context=config.d_model,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout,
        )

        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            input_dim=config.d_model,
            output_dim=config.d_model,
            up_proj_dim=config.ffn_up_dim,
            hidden_dropout=config.ffn_dropout,
            output_dropout=config.ffn_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope: RoPE2D,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.self_attn_norm(x),
            rotary_emb=rope,
        )

        x = x + self.cross_attn(
            x=self.cross_attn_norm(x),
            context=context,
            rotary_emb=rope,
        )

        x = x + self.ffn(
            self.ffn_norm(x),
        )

        return x


class ValueHead(nn.Module):
    """
    Critic Value Head。

    输入:
        enc_out: (B, N+1, d_model)

    输出:
        value: (B, 1)
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        layers = [
            nn.Linear(config.d_model, config.value_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.value_hidden_dim, 1),
        ]

        if config.value_use_tanh:
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        # enc_out shape: (B, 1 + N, C)
        # 提取 [CLS] token 的特征，它总是位于索引 0
        cls_feat = enc_out[:, 0, :]  # (B, C)

        # MLP 输出 Value
        value = self.net(cls_feat)

        return value


class PolicyHead(nn.Module):
    """
    Actor Policy Head。

    输入:
        dec_out: (B, N, d_model)

    输出:
        logits: (B, N)
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        layers = [
            nn.Linear(config.d_model, config.policy_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.policy_hidden_dim, 1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, dec_out: torch.Tensor) -> torch.Tensor:

        logits = self.net(dec_out).squeeze(-1)

        return logits


class GomokuTransformer(nn.Module):
    """
    五子棋 Actor-Critic Transformer。

    输入:
        state:
            shape = (B, 2, 15, 15)

        valid_mask:
            shape = (B, 225)
            dtype = torch.bool
            True 表示可以落子，False 表示非法动作。

    输出:
        policy_logits:
            shape = (B, 225)

        value:
            shape = (B, 1)
    """

    def __init__(self, config: GomokuConfig):
        super().__init__()

        self.config = config

        self.tokenizer = BoardCNNTokenizer(config)

        self.rope2d = RoPE2D(
            head_dim=config.head_dim,
            height=config.board_size,
            width=config.board_size,
        )

        # [CLS] token / 可学习
        # shape: (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.encoders = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.encoder_layers)]
        )
        self.encoder_norm = RMSNorm(config.d_model)

        self.decoders = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.decoder_layers)]
        )
        self.decoder_norm = RMSNorm(config.d_model)

        self.value_head = ValueHead(config)
        self.policy_head = PolicyHead(config)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        更完整的初始化：
        - Linear: 正态初始化
        - Conv2d: Kaiming 初始化，更适合 CNN + SiLU
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (B, N, C)

        Returns:
            encoder (B, N, C)
        """
        B = tokens.shape[0]

        # 拼接 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, C)
        enc_out = torch.cat((cls_tokens, tokens), dim=1)

        for block in self.encoders:
            enc_out = block(enc_out, self.rope2d)

        enc_out = self.encoder_norm(enc_out)

        return enc_out

    def decode(
        self,
        tokens: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码 policy 特征。

        Returns:
            dec_out: (B, N, C)
        """
        dec_out = tokens

        for block in self.decoders:
            dec_out = block(
                x=dec_out,
                context=enc_out,
                rope=self.rope2d,
            )

        dec_out = self.decoder_norm(dec_out)

        return dec_out

    def apply_action_mask(
        self,
        logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        对非法动作进行 mask。

        Args:
            logits:
                shape = (B, N)

            valid_mask:
                shape = (B, N), bool

        Returns:
            masked_logits:
                shape = (B, N)
        """
        if valid_mask is None:
            return logits

        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask.bool()

        if valid_mask.shape != logits.shape:
            raise ValueError(
                f"valid_mask shape must be same as logits shape, "
                f"got valid_mask={valid_mask.shape}, logits={logits.shape}"
            )

        # 正常情况下不应该出现全 False。
        # 如果出现，Categorical 会出问题，所以提前报错更好。
        if not valid_mask.any(dim=-1).all():
            raise ValueError("Each sample must have at least one valid action.")

        logits = logits.masked_fill(
            ~valid_mask,
            self.config.invalid_logit,
        )

        return logits

    def forward(
        self,
        state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state:
                shape = (B, 2, board_size, board_size)

            valid_mask:
                shape = (B, board_size * board_size)

            return_features:
                如果为 True，额外返回中间特征字典。

        Returns:
            policy_logits:
                shape = (B, board_size * board_size)

            value:
                shape = (B, 1)

            features: optional dict
        """
        if state.dim() != 4:
            raise ValueError(
                f"state must be 4D tensor: (B, C, H, W), got shape={state.shape}"
            )

        b, c, h, w = state.shape

        if c != self.config.in_channels:
            raise ValueError(
                f"state channel mismatch, expected {self.config.in_channels}, got {c}"
            )

        if h != self.config.board_size or w != self.config.board_size:
            raise ValueError(
                f"board size mismatch, expected "
                f"{self.config.board_size}x{self.config.board_size}, got {h}x{w}"
            )

        tokens = self.tokenizer(state)

        enc_out = self.encode(tokens)

        value = self.value_head(enc_out)

        dec_out = self.decode(
            tokens=tokens,
            enc_out=enc_out,
        )

        policy_logits = self.policy_head(dec_out)

        policy_logits = self.apply_action_mask(
            logits=policy_logits,
            valid_mask=valid_mask,
        )

        if return_features:
            features: Dict[str, Any] = {
                "tokens": tokens,
                "enc_out": enc_out,
                "dec_out": dec_out,
            }
            return policy_logits, value, features

        return policy_logits, value

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        方便 Agent 调用的动作选择函数。

        Args:
            state:
                shape = (B, 2, board_size, board_size)

            valid_mask:
                shape = (B, board_size * board_size)

            deterministic:
                True 则选择 argmax，False 则采样。

            temperature:
                采样温度。越小越贪心。

        Returns:
            action:
                shape = (B,)

            log_prob:
                shape = (B,)

            value:
                shape = (B,)
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        logits, value = self.forward(
            state=state,
            valid_mask=valid_mask,
        )

        logits = logits / temperature

        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    def get_action_distribution(
        self,
        state: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        PPO 更新时很方便：

            dist, value = model.get_action_distribution(...)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

        Returns:
            dist:
                Categorical distribution

            value:
                shape = (B,)
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        logits, value = self.forward(
            state=state,
            valid_mask=valid_mask,
        )

        logits = logits / temperature

        dist = Categorical(logits=logits)

        return dist, value.squeeze(-1)


if __name__ == "__main__":
    config = GomokuConfig(
        board_size=15,
        in_channels=2,
        )

    model = GomokuTransformer(config)

    batch_size = 2

    state = torch.zeros(
        batch_size,
        2,
        config.board_size,
        config.board_size,
        dtype=torch.float32,
    )

    valid_mask = torch.ones(
        batch_size,
        config.seq_len,
        dtype=torch.bool,
    )

    logits, value = model(state, valid_mask)

    print("logits:", logits.shape)
    print("value:", value.shape)

    action, log_prob, value = model.act(
        state,
        valid_mask,
        deterministic=False,
    )

    print("action:", action)
    print("log_prob:", log_prob)
    print("value:", value)
