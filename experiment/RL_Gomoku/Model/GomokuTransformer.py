import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from experiment.RL_Gomoku.Model.rmsNorm import RMSNorm
from experiment.RL_Gomoku.Model.RoPE2D import RoPE2D
from experiment.RL_Gomoku.Model.Base.baseSelfAttention import SelfAttention
from experiment.RL_Gomoku.Model.Base.baseCrossAttention import CrossAttention
from experiment.RL_Gomoku.Model.Base.baseSwiGLU import SwiGLU


# ==========================================
# 1. 模型配置类 (极度方便后期扩展和修改超参数)
# ==========================================
@dataclass
class GomokuConfig:
    board_size: int = 15  # 五子棋盘大小 15x15
    in_channels: int = 2  # 输入 2 个通道 (己方，对方)
    d_model: int = 128  # 隐藏层维度
    num_heads: int = 4  # 注意力头数
    encoder_layers: int = 3  # 编码器层数 (提取全局 Value 特征)
    decoder_layers: int = 3  # 解码器层数 (计算局部 Policy 策略)
    ffn_up_dim: int = 342  # SwiGLU 升维大小 (通常是 8/3 * d_model，这里取近值)

    # 强化学习 PPO 专用设置：坚决不用 Dropout
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    ffn_dropout: float = 0.0

    @property
    def seq_len(self) -> int:
        return self.board_size * self.board_size


# ==========================================
# 专门为 2D 棋盘设计的 CNN 特征提取器
# ==========================================
class BoardCNNTokenizer(nn.Module):
    def __init__(self, config: GomokuConfig):
        super().__init__()
        hidden_dim = config.d_model // 2

        self.conv1 = nn.Conv2d(config.in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        # GroupNorm(1, C) 等价于在 C 维度做 LayerNorm，天然支持 4D 张量
        self.norm1 = nn.GroupNorm(1, hidden_dim)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(hidden_dim, config.d_model, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, config.d_model)
        self.act2 = nn.SiLU()

        self.final_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2).contiguous()
        return self.final_norm(x)


# ==========================================
# 2. 核心模块构建：Encoder 层与 Decoder 层
# ==========================================
class EncoderLayer(nn.Module):
    """单层 Encoder：负责自注意力融合，提取棋盘全局特征"""

    def __init__(self, config: GomokuConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.proj_dropout
        )
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            input_dim=config.d_model,
            output_dim=config.d_model,
            up_proj_dim=config.ffn_up_dim,
            hidden_dropout=config.ffn_dropout,
            output_dropout=config.ffn_dropout
        )

    def forward(self, x: torch.Tensor, rope: RoPE2D) -> torch.Tensor:
        # Pre-Norm 架构 (LLaMA 标配，训练极稳定)
        x = x + self.attn(self.norm1(x), rotary_emb=rope)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """单层 Decoder：负责根据 Encoder 传来的大局观，结合自身位置计算落子策略"""

    def __init__(self, config: GomokuConfig):
        super().__init__()
        # Decoder 自身的 Self-Attention
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(
            d_model=config.d_model, num_heads=config.num_heads,
            attn_dropout=config.attn_dropout, proj_dropout=config.proj_dropout
        )
        # 与 Encoder 交互的 Cross-Attention
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(
            d_model=config.d_model, num_heads=config.num_heads, d_context=config.d_model,
            attn_dropout=config.attn_dropout, proj_dropout=config.proj_dropout
        )
        # 前馈网络
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(
            input_dim=config.d_model, output_dim=config.d_model, up_proj_dim=config.ffn_up_dim,
            hidden_dropout=config.ffn_dropout, output_dropout=config.ffn_dropout
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, rope: RoPE2D) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), rotary_emb=rope)
        x = x + self.cross_attn(x=self.norm2(x), context=context, rotary_emb=rope)
        x = x + self.ffn(self.norm3(x))
        return x


class GomokuTransformer(nn.Module):
    def __init__(self, config: GomokuConfig):
        super().__init__()
        self.config = config

        # 1. CNN Tokenizer
        self.tokenizer = BoardCNNTokenizer(config)

        # 2. RoPE2D
        self.rope2d = RoPE2D(
            head_dim=config.d_model // config.num_heads,
            height=config.board_size,
            width=config.board_size
        )

        # 3. Encoders
        self.encoders = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.encoder_layers)
        ])
        self.encoder_norm = RMSNorm(config.d_model)

        # 4. Decoders
        self.decoders = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.decoder_layers)
        ])
        self.decoder_norm = RMSNorm(config.d_model)

        # 5. Value (Critic)
        self.value_proj = nn.Linear(config.d_model, 2)
        flatten_dim = config.seq_len * 2
        self.value_mlp = nn.Sequential(
            nn.Linear(flatten_dim, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 1, bias=False),
            nn.Tanh()
        )

        # 6. Policy Head
        self.policy_head = nn.Linear(config.d_model, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        valid_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, 2, 15, 15) 你的环境输出的张量
            valid_mask: (B, 225) boolean张量，True表示合法动作。用于 PPO 动作屏蔽。
        Returns:
            policy_logits: (B, 225)
            value: (B, 1)
        """
        # --- A. CNN 提取特征 ---
        # (B, 2, 15, 15) -> (B, 225, d_model)
        x = self.tokenizer(state)

        # --- B. 编码器全局融合 ---
        enc_out = x
        for encoder in self.encoders:
            enc_out = encoder(enc_out, self.rope2d)
        enc_out = self.encoder_norm(enc_out)

        # --- C. 计算 Value ---
        # 1. 通道降维: (B, 225, d_model) -> (B, 225, 2)
        v_feat = self.value_proj(enc_out)
        # 2. 展平空间: (B, 225, 2) -> (B, 450)
        v_feat = v_feat.view(v_feat.size(0), -1)
        # 3. MLP 输出: (B, 450) -> (B, 1)
        value = self.value_mlp(v_feat)

        # --- D. 解码器交叉注意力 ---
        dec_out = x
        for decoder in self.decoders:
            dec_out = decoder(dec_out, context=enc_out, rope=self.rope2d)
        dec_out = self.decoder_norm(dec_out)

        # --- E. 计算 Policy ---
        # (B, 225, 1) -> (B, 225)
        policy_logits = self.policy_head(dec_out).squeeze(-1)

        # --- F. 合法动作屏蔽 (Action Masking) ---
        # 如果传入了 mask，把不能下的地方设为极其小的负数，Softmax 之后概率就是 0
        if valid_mask is not None:
            # PPO 训练中，强烈建议在这里直接屏蔽掉无效动作！
            policy_logits = policy_logits.masked_fill(~valid_mask, -1e9)

        return policy_logits, value


if __name__ == "__main__":
    from torchinfo import summary
    from experiment.RL_Gomoku.Env.GomokuEnv import GomokuEnv

    env = GomokuEnv()

    # 实例化网络
    config = GomokuConfig()
    model = GomokuTransformer(config)


    # 获取环境初始状态
    state, info = env.reset()

    # 转换为 Tensor，并增加 Batch 维度
    # state 形状: (2, 15, 15) -> (1, 2, 15, 15)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # 获取 valid_mask 并转换为 Tensor
    valid_mask = torch.tensor(info["valid_mask"], dtype=torch.bool).unsqueeze(0)

    # 前向传播 (带上 Mask)
    policy_logits, value = model(state_tensor, valid_mask=valid_mask)

    # 计算落子概率分布
    import torch.nn.functional as F

    probs = F.softmax(policy_logits, dim=-1)

    print("Value 胜率评估:", value.item())
    print("当前无法落子的地方的 Logits (应为 -1e9):", policy_logits[0, 0].item() if not valid_mask[0, 0] else "可落子")

    # 根据网络采样一个合法动作
    m = torch.distributions.Categorical(probs)
    action = m.sample().item()

    print(f"网络选择的落子点 (1D 索引): {action}")
    print(f"坐标: {divmod(action, 15)}")

    summary(model, input_size=(2, 2, 15, 15), device="cpu")

