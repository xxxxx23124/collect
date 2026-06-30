from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint



@dataclass
class TransUNetConfig:
    image_size: int | tuple[int, int] = 512
    time_emb_dim: int = 128
    time_hidden_dim: int = 256
    time_layers: int = 3
    num_experts: int = 2
    attn_head_dim: int = 32
    rope_base: float = 500.0
    rope_max_size: int | tuple[int, int] | None = None
    groups: int = 4
    dense_inc: int = 8
    stem_channels: int = 32
    encoder_channels: tuple[int, ...] = (32, 64, 112)
    encoder_blocks: tuple[int, ...] = (1, 1, 2)
    decoder_blocks: tuple[int, ...] = (2, 1, 1)
    decoder_channels: tuple[int, ...] | None = None
    bottleneck_channels: int = 256
    bottleneck_inner_channels: int = 256
    bottleneck_layers: int = 2
    bottleneck_heads: int = 8
    encoder_attn_levels: tuple[int, ...] = (2,)
    decoder_attn_levels: tuple[int, ...] = (0,)
    encoder_attn_heads: int = 4
    decoder_attn_heads: int = 4
    ortho_weight: float = 1e-6
    router_temperature: float = 1.0
    router_balance_weight: float = 1e-6
    router_entropy_weight: float = 0.0
    collect_router_auxiliary_loss: bool = True


def _to_2tuple(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or a tuple of two ints, got {value!r}.")


class Factory:
    def __init__(self,
                 time_emb_dim,
                 convolution_kernel_group=4,
                 attn_head_dim=64,
                 rope_height=256,
                 rope_width=256,
                 rope_base=500.0,
                 router_temperature=1.0,
                 collect_router_auxiliary_loss=True,):
        self.time_emb_dim = time_emb_dim
        self.convolution_kernel_group = convolution_kernel_group
        self.attn_head_dim = attn_head_dim
        self.router_temperature = router_temperature
        self.collect_router_auxiliary_loss = collect_router_auxiliary_loss
        self.rope = RoPE2D(
            dim=self.attn_head_dim,
            height=rope_height,
            width=rope_width,
            base=rope_base
        )

    def get_condconv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        return TimeAwareCondConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            time_emb_dim=self.time_emb_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            num_experts=self.convolution_kernel_group,
            router_temperature=self.router_temperature,
            collect_router_auxiliary_loss=self.collect_router_auxiliary_loss
        )

    def get_time_mlp(self, hidden_dim=512, num_layers=5, fourier_scale=16.0):
        return TimeMLP(
            time_emb_dim=self.time_emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            fourier_scale=fourier_scale
        )

    def get_act(self):
        return nn.SiLU()

    def get_cln(self, channels):
        return AdaCLN(
            channels=channels,
            time_emb_dim=self.time_emb_dim
        )

    def get_grn(self, channels):
        return AdaGRN(
            channels=channels,
            time_emb_dim=self.time_emb_dim
        )

    def get_rms(self, channels):
        return AdaRMSNorm(
            channels=channels,
            time_emb_dim=self.time_emb_dim
        )

    def get_selfattn(self, channels, num_heads):
        return SelfAttention(
            channels=channels,
            num_heads=num_heads,
            rope=self.rope
        )

    def get_crossattn(self, channels, num_heads):
        return CrossAttention(
            channels=channels,
            num_heads=num_heads,
            rope=self.rope
        )

    def get_swiglu(self, channels, inner_channels):
        return TimeAwareSwiGLU(
            channels=channels,
            inner_channels=inner_channels,
            time_emb_dim=self.time_emb_dim
        )


class RoPE2D(nn.Module):
    def __init__(
            self,
            dim: int = 64,
            height: int = 256,
            width: int = 256,
            base: float = 500.0
    ):
        super().__init__()

        assert dim % 4 == 0, f"dim must be 4*n, got 4*{dim // 4}+{dim % 4}"

        self.dim = dim
        self.height = height
        self.width = width
        self.base = base

        half_dim = dim // 2
        quarter_dim = half_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, dtype=torch.float) / quarter_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _build_freqs(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        if height > self.height or width > self.width:
            raise ValueError(
                f"RoPE2D cache supports up to {(self.height, self.width)}, got {(height, width)}."
            )

        half_dim = self.dim // 2
        inv_freq = self.inv_freq.to(device=device)

        t_y = torch.arange(height, device=device, dtype=inv_freq.dtype)
        freqs_y = torch.einsum('i,j->ij', t_y, inv_freq)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)
        emb_y = emb_y.unsqueeze(1).expand(-1, width, -1).reshape(1, 1, height * width, half_dim)

        t_x = torch.arange(width, device=device, dtype=inv_freq.dtype)
        freqs_x = torch.einsum('i,j->ij', t_x, inv_freq)
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
        emb_x = emb_x.unsqueeze(0).expand(height, -1, -1).reshape(1, 1, height * width, half_dim)

        return emb_y.cos().to(dtype), emb_y.sin().to(dtype), emb_x.cos().to(dtype), emb_x.sin().to(dtype)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                height: int | None = None,
                width: int | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]

        if height is None or width is None:
            side = int(math.sqrt(seq_len))
            if side * side != seq_len:
                raise ValueError("RoPE2D needs height and width for non-square sequences.")
            height = width = side

        if seq_len != height * width:
            raise ValueError(f"RoPE2D got seq_len={seq_len}, but height*width={height * width}.")

        cos_y, sin_y, cos_x, sin_x = self._build_freqs(height, width, q.device, q.dtype)

        def main_path(x_in: torch.Tensor):
            y_part, x_part = x_in.chunk(2, dim=-1)

            y_out = y_part * cos_y + self.rotate_half(y_part) * sin_y
            x_out = x_part * cos_x + self.rotate_half(x_part) * sin_x

            return torch.cat((y_out, x_out), dim=-1)

        q_embed = main_path(q)
        k_embed = main_path(k)

        return q_embed, k_embed

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=16.0):
        super().__init__()
        W = torch.randn(embed_dim // 2) * scale
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TimeMLP(nn.Module):
    def __init__(self,
                 time_emb_dim=256,
                 hidden_dim=512,
                 num_layers=6,
                 fourier_scale=16.0):
        super().__init__()
        self.fourier = GaussianFourierProjection(hidden_dim, scale=fourier_scale)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.mapping = nn.ModuleList([
            ResMLPBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, time_emb_dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.dtype == torch.long:
            time = time.float()
        x = self.fourier(time)
        x = self.input_proj(x)
        for block in self.mapping:
            x = block(x)
        return self.out_proj(x)


class Modulator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        hidden_dim = max(64, in_channels // 4)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_channels),
        )
        self.act = nn.SiLU()
        self.to_out = nn.Linear(in_channels, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_out.weight)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        time_emb = self.act(time_emb + self.net(time_emb))
        return self.to_out(time_emb)


class TinyContentEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=None, eps=1e-6):
        super().__init__()
        out_channels = out_channels if out_channels is not None else min(64, max(16, in_channels // 4))
        self.feat_channels = out_channels
        self.out_channels = out_channels * 2
        self.eps = eps
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        feat_mean = feat.mean(dim=(2, 3))
        feat_std = feat.std(dim=(2, 3), unbiased=False) + self.eps
        return torch.cat([feat_mean, feat_std], dim=1)


class ContentAwareModulator(nn.Module):
    def __init__(self,
                 in_channels,
                 time_emb_dim,
                 out_channels,
                 hidden_dim=None,
                 eps=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(64, in_channels // 4)
        self.eps = eps
        self.content_encoder = TinyContentEncoder(in_channels)
        content_dim = in_channels * 2 + self.content_encoder.out_channels
        self.to_q = nn.Linear(time_emb_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(content_dim, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(content_dim, self.hidden_dim, bias=False)
        self.mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.to_out = nn.Linear(self.hidden_dim, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_out.weight)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False) + self.eps
        tiny_feat = self.content_encoder(x)
        # 均值/方差提供稳定全局统计，小卷积补充局部纹理信息。
        return torch.cat([mu, std, tiny_feat], dim=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        stats = self._compute_stats(x)
        q = self.to_q(time_emb)
        k = self.to_k(stats)
        v = self.to_v(stats)

        attn = torch.sigmoid(q * k)
        x_hidden = v * attn
        x_mixed = x_hidden + self.mixer(x_hidden)
        return self.to_out(x_mixed)


class CondConvRouter(nn.Module):
    def __init__(self,
                 channels,
                 time_emb_dim,
                 num_experts=4,
                 hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modulator = ContentAwareModulator(
            in_channels=channels,
            time_emb_dim=time_emb_dim,
            out_channels=num_experts,
            hidden_dim=hidden_dim
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return self.modulator(x, time_emb)


class TimeAwareCondConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 bias: bool = False,
                 num_experts: int = 4,
                 gating_noise_std: float = 1e-2,
                 router_temperature: float = 1.0,
                 collect_router_auxiliary_loss: bool = True, ):
        super().__init__()

        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts
        self.gating_noise_std = gating_noise_std
        self.router_temperature = router_temperature
        self.collect_router_auxiliary_loss = collect_router_auxiliary_loss

        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive.")

        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        self.router = CondConvRouter(in_channels, time_emb_dim, num_experts)
        ortho_mask = torch.ones(num_experts, num_experts) - torch.eye(num_experts)
        self.register_buffer("ortho_offdiag_mask", ortho_mask)
        self._last_routing_weights: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self):
        fan_in = (self.in_channels // self.groups) * (self.kernel_size ** 2)

        std = math.sqrt(2.0 / fan_in)

        nn.init.normal_(self.weight, mean=0.0, std=std)

        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def get_ortho_loss(self):
        w = self.weight
        n_exp, out_c = w.shape[:2]

        # 逐输出通道约束专家方向不同，只惩罚专家间的非对角相似度。
        w_per_channel = w.permute(1, 0, 2, 3, 4).reshape(out_c, n_exp, -1)
        w_norm = F.normalize(w_per_channel, p=2, dim=2, eps=1e-6)
        gram = torch.bmm(w_norm, w_norm.transpose(1, 2).contiguous())
        off_diag = gram * self.ortho_offdiag_mask.to(dtype=gram.dtype, device=gram.device)

        return off_diag.square().mean()

    def set_router_auxiliary_loss_enabled(self, enabled: bool):
        self.collect_router_auxiliary_loss = bool(enabled)
        if not self.collect_router_auxiliary_loss:
            self._last_routing_weights = None

    def get_router_loss(self, balance_weight=1e-6, entropy_weight=0.0):
        routing_weights = self._last_routing_weights
        if routing_weights is None:
            return None

        loss = routing_weights.new_zeros(())

        if balance_weight > 0:
            avg_prob = routing_weights.mean(dim=0)
            target = torch.full_like(avg_prob, 1.0 / self.num_experts)
            loss = loss + balance_weight * F.mse_loss(avg_prob, target)

        if entropy_weight > 0:
            probs = routing_weights.clamp_min(1e-8)
            entropy = -(probs * probs.log()).sum(dim=1).mean()
            loss = loss + entropy_weight * (math.log(self.num_experts) - entropy)

        return loss

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        routing_logits = self.router(x, time_emb)
        routing_weights = F.softmax(routing_logits / self.router_temperature, dim=1)
        if self.training and self.collect_router_auxiliary_loss:
            self._last_routing_weights = routing_weights
        else:
            self._last_routing_weights = None

        out = None
        for expert_idx in range(self.num_experts):
            expert_bias = self.bias[expert_idx] if self.bias is not None else None
            expert_out = F.conv2d(
                x,
                self.weight[expert_idx],
                expert_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=1,
                groups=self.groups
            )
            expert_out = expert_out * routing_weights[:, expert_idx].view(-1, 1, 1, 1)
            out = expert_out if out is None else out + expert_out

        return out


class AdaCLN(nn.Module):
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.num_channels = channels
        self.eps = eps
        self.emb_layers = Modulator(time_emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)

        emb = self.emb_layers(time_emb)
        gamma, beta = emb.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x_norm * (1 + gamma) + beta


class AdaGRN(nn.Module):
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.modulator = ContentAwareModulator(
            in_channels=channels,
            time_emb_dim=time_emb_dim,
            out_channels=2 * channels,
            hidden_dim=max(64, channels // 4)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        params = self.modulator(x, time_emb)
        gamma, beta = params.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x * (1 + gamma * nx) + beta


class AdaRMSNorm(nn.Module):
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = channels
        self.time_proj = Modulator(time_emb_dim, channels)
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_x = norm_x * self.scale
        time_scale = self.time_proj(time_emb).unsqueeze(1)
        out = norm_x * (1 + time_scale)

        return out


class SelfAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 rope: RoPE2D):
        super().__init__()
        self.num_heads = num_heads
        head_dim = rope.dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, channels, bias=False)

        self.rope = rope

    def forward(self,
                x: torch.Tensor,
                height: int | None = None,
                width: int | None = None) -> torch.Tensor:
        B, N, C = x.shape

        if height is None or width is None:
            side = int(math.sqrt(N))
            if side * side != N:
                raise ValueError("SelfAttention needs height and width for non-square sequences.")
            height = width = side

        if N != height * width:
            raise ValueError(f"SelfAttention got N={N}, but height*width={height * width}.")

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k, height=height, width=width)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 rope: RoPE2D):
        super().__init__()
        self.num_heads = num_heads
        head_dim = rope.dim
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(channels, inner_dim, bias=False)
        self.to_v = nn.Linear(channels, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, channels, bias=False)

        self.rope = rope
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_out.weight)

    def forward(self,
                query: torch.Tensor,
                context: torch.Tensor,
                height: int | None = None,
                width: int | None = None) -> torch.Tensor:
        B, Nq, _ = query.shape
        _, Nk, _ = context.shape

        if height is None or width is None:
            side = int(math.sqrt(Nq))
            if side * side != Nq:
                raise ValueError("CrossAttention needs height and width for non-square sequences.")
            height = width = side

        if Nq != height * width or Nk != height * width:
            raise ValueError(
                f"CrossAttention got Nq={Nq}, Nk={Nk}, but height*width={height * width}."
            )

        q = self.to_q(query)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k, height=height, width=width)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.to_out(out)


class TimeAwareSwiGLU(nn.Module):
    def __init__(self,
                 channels: int,
                 inner_channels: int,
                 time_emb_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(channels, inner_channels, bias=False)
        self.w_val = nn.Linear(channels, inner_channels, bias=False)
        self.w_time = Modulator(time_emb_dim, inner_channels)
        self.w_out = nn.Linear(inner_channels, channels, bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        c_gate = self.w_gate(x)
        val = self.w_val(x)
        t_gate = self.w_time(time_emb).unsqueeze(1)
        gate = self.act(c_gate * (1 + t_gate))
        out = self.w_out(gate * val)
        return out


class DualPathBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 res_channels: int,
                 dense_inc: int,
                 groups: int,
                 factory: Factory,
                 expansion_ratio: int = 4, ):
        super().__init__()
        assert res_channels % groups == 0, f"res_channels:{res_channels} must be divisible by groups:{groups}."
        self.res_channels = res_channels
        self.conv1 = factory.get_condconv(in_channels=in_channels,
                                          out_channels=res_channels,
                                          kernel_size=1,
                                          bias=False)

        self.conv2 = factory.get_condconv(in_channels=res_channels,
                                          out_channels=res_channels,
                                          kernel_size=7,
                                          padding=3,
                                          groups=groups,
                                          bias=False)

        self.cln = factory.get_cln(res_channels)

        mid_channels = res_channels * expansion_ratio
        self.conv3_up = factory.get_condconv(in_channels=res_channels,
                                             out_channels=mid_channels,
                                             kernel_size=1,
                                             bias=False)

        self.act = factory.get_act()

        self.grn = factory.get_grn(mid_channels)
        self.conv3_res = factory.get_condconv(in_channels=mid_channels,
                                              out_channels=res_channels,
                                              kernel_size=1,
                                              bias=False)
        self.conv3_dense = factory.get_condconv(in_channels=mid_channels,
                                                out_channels=dense_inc,
                                                kernel_size=1,
                                                bias=False)

        self.gamma_res = nn.Parameter(1e-6 * torch.ones(res_channels), requires_grad=True)

        self.gamma_dense = nn.Parameter(1e-2 * torch.ones(dense_inc), requires_grad=True)

    def forward(self, x, time_emb) -> torch.Tensor:
        def main_path(x_in: torch.Tensor, time_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.conv1(x_in, time_emb)
            out = self.conv2(out, time_emb)
            out = self.cln(out, time_emb)
            out = self.conv3_up(out, time_emb)
            out = self.act(out)
            out = self.grn(out, time_emb)
            out_res = self.conv3_res(out, time_emb)
            out_dense = self.conv3_dense(out, time_emb)
            return out_res, out_dense

        new_res, new_dense = main_path(x, time_emb)

        prev_res = x[:, :self.res_channels, :, :]
        prev_dense = x[:, self.res_channels:, :, :]
        residual = prev_res + new_res * self.gamma_res.view(1, -1, 1, 1)
        if prev_dense.size(1) > 0:
            dense = torch.cat((prev_dense, new_dense * self.gamma_dense.view(1, -1, 1, 1)), dim=1)
        else:
            dense = new_dense * self.gamma_dense.view(1, -1, 1, 1)

        return torch.cat((residual, dense), dim=1)


class DualPathStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 res_channels: int,
                 num_blocks: int,
                 dense_inc: int,
                 groups: int,
                 factory: Factory,
                 expansion_ratio: int = 4):
        super().__init__()

        initial_dense = in_channels - res_channels
        assert initial_dense >= 0, "res_channels cannot be greater than in_channels."

        current_channels = in_channels

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = DualPathBlock(
                in_channels=current_channels,
                res_channels=res_channels,
                dense_inc=dense_inc,
                groups=groups,
                factory=factory,
                expansion_ratio=expansion_ratio
            )
            self.blocks.append(block)
            current_channels += dense_inc

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor) -> torch.Tensor:

            for block in self.blocks:
                x_in = block(x_in, time_emb_in)
            return x_in

        if self.training:
            return checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            return main_path(x, time_emb)


class DownsampleLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 factory: Factory, ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_inc = dense_inc

        self.cln = factory.get_cln(in_channels)
        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=out_channels + dense_inc,
            kernel_size=2,
            stride=2,
            padding=0,
            groups=1,
            bias=False
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.cln(x, time_emb)
        return self.conv(x, time_emb)


class PixelShuffleUpsampleLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 factory: Factory, ):
        super().__init__()
        target_out = out_channels + dense_inc
        expansion = 4

        inner_out = target_out * expansion

        self.cln = factory.get_cln(in_channels)
        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=inner_out,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.conv, TimeAwareCondConv2d):
            for i in range(self.conv.num_experts):
                self._icnr_init(self.conv.weight[i], scale=2)

    def _icnr_init(self, tensor, scale=2):
        out_c, in_c, k, k = tensor.shape

        assert out_c % (scale ** 2) == 0, f"out_c % (scale ** 2) should be 0, now is {out_c % (scale ** 2)}"

        new_out_c = out_c // (scale ** 2)
        kernel = torch.randn(new_out_c, in_c, k, k) * 0.1
        kernel = kernel.repeat_interleave(scale ** 2, dim=0)

        with torch.no_grad():
            tensor.copy_(kernel)

    def forward(self, x, time_emb) -> torch.Tensor:
        x = self.cln(x, time_emb)
        x = self.conv(x, time_emb)
        x = self.pixel_shuffle(x)
        return x


class FeatureFusionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 factory: Factory, ):
        super().__init__()
        self.cln = factory.get_cln(in_channels)
        self.act = factory.get_act()
        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.cln(x, time_emb)
        x = self.act(x)
        return self.conv(x, time_emb)


class SpatialSelfAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 factory: Factory, ):
        super().__init__()
        self.num_heads = num_heads
        self.rms = factory.get_rms(
            channels=channels,
        )
        self.attn = factory.get_selfattn(
            channels=channels,
            num_heads=num_heads,
        )
        self.layer_scale = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        residual = x
        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        x_norm = self.rms(x_flat, time_emb)
        out = self.attn(x_norm, height=H, width=W)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return residual + out * self.layer_scale.view(1, C, 1, 1)


class TimeAwareIdentity(nn.Module):
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        return x


class AttentiveDualPathStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 res_channels: int,
                 num_blocks: int,
                 dense_inc: int,
                 groups: int,
                 factory: Factory,
                 attn_heads: int | None = None,
                 expansion_ratio: int = 4):
        super().__init__()
        self.stage = DualPathStage(
            in_channels=in_channels,
            res_channels=res_channels,
            num_blocks=num_blocks,
            dense_inc=dense_inc,
            groups=groups,
            factory=factory,
            expansion_ratio=expansion_ratio
        )
        self.out_channels = self.stage.out_channels
        self.attn = (
            SpatialSelfAttention(self.out_channels, attn_heads, factory)
            if attn_heads is not None
            else TimeAwareIdentity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.stage(x, time_emb)
        return self.attn(x, time_emb)


class TransformerBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 factory: Factory,
                 num_heads: int = 8,
                 ffn_mult: float = 4.0, ):
        super().__init__()
        self.num_heads = num_heads

        self.rms1 = factory.get_rms(channels=channels)

        self.attn = factory.get_selfattn(channels=channels, num_heads=num_heads)
        self.attn_scale = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)

        self.rms2 = factory.get_rms(channels=channels)

        inner_channels = int(channels * ffn_mult)
        self.ffn = factory.get_swiglu(channels=channels, inner_channels=inner_channels)
        self.ffn_scale = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)

    def forward(self,
                x: torch.Tensor,
                time_emb: torch.Tensor,
                height: int | None = None,
                width: int | None = None) -> torch.Tensor:

        x = x + self.attn(self.rms1(x, time_emb), height=height, width=width) * self.attn_scale.view(1, 1, -1)
        x = x + self.ffn(self.rms2(x, time_emb), time_emb) * self.ffn_scale.view(1, 1, -1)
        return x


class CrossFusionBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 factory: Factory,
                 gate_init_bias: float = -2.0):
        super().__init__()
        self.q_norm = factory.get_rms(channels=channels)
        self.kv_norm = factory.get_rms(channels=channels)
        self.cross_attn = factory.get_crossattn(channels=channels, num_heads=num_heads)
        self.cross_grn = factory.get_grn(channels=channels)
        self.gate = ContentAwareModulator(
            in_channels=channels,
            time_emb_dim=factory.time_emb_dim,
            out_channels=channels,
            hidden_dim=max(64, channels // 4)
        )
        self.gate_init_bias = gate_init_bias
        self.layer_scale = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)
        self.query_mix = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self,
                cnn_tokens: torch.Tensor,
                state_tokens: torch.Tensor,
                time_emb: torch.Tensor,
                height: int,
                width: int) -> torch.Tensor:
        B, N, C = state_tokens.shape
        query_tokens = cnn_tokens + self.query_mix.view(1, 1, C) * (state_tokens - cnn_tokens)
        query_map = query_tokens.transpose(1, 2).reshape(B, C, height, width)
        cross_tokens = self.cross_attn(
            self.q_norm(query_tokens, time_emb),
            self.kv_norm(state_tokens, time_emb),
            height=height,
            width=width
        )

        cross_map = cross_tokens.transpose(1, 2).reshape(B, C, height, width)
        cross_map = self.cross_grn(cross_map, time_emb)
        gate = torch.sigmoid(self.gate(query_map, time_emb) + self.gate_init_bias)

        state_map = state_tokens.transpose(1, 2).reshape(B, C, height, width)
        state_map = state_map + gate.view(B, C, 1, 1) * self.layer_scale.view(1, C, 1, 1) * cross_map
        return state_map.flatten(2).transpose(1, 2).contiguous()


class BottleneckTransformerStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 inner_channels: int,
                 num_layers: int,
                 num_heads: int,
                 factory: Factory,
                 ffn_mult: float = 4.0, ):
        super().__init__()

        self.proj_in = factory.get_condconv(
            in_channels=in_channels,
            out_channels=inner_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                channels=inner_channels,
                factory=factory,
                num_heads=num_heads,
                ffn_mult=ffn_mult
            )
            for _ in range(num_layers)
        ])

        self.cross_layers = nn.ModuleList([
            CrossFusionBlock(
                channels=inner_channels,
                num_heads=num_heads,
                factory=factory
            )
            for _ in range(num_layers)
        ])

        self.proj_out = factory.get_condconv(
            in_channels=inner_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

        self.trans_scale = nn.Parameter(1e-2 * torch.ones(inner_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor) -> torch.Tensor:
            x_feat = self.proj_in(x_in, time_emb_in)
            B, C, H, W = x_feat.shape
            cnn_tokens = x_feat.flatten(2).transpose(1, 2).contiguous()
            state_tokens = cnn_tokens

            for layer, cross_layer in zip(self.layers, self.cross_layers):
                state_tokens = layer(state_tokens, time_emb_in, height=H, width=W)
                state_tokens = cross_layer(
                    cnn_tokens=cnn_tokens,
                    state_tokens=state_tokens,
                    time_emb=time_emb_in,
                    height=H,
                    width=W
                )

            out = state_tokens.transpose(1, 2).reshape(B, C, H, W)
            out = x_feat + self.trans_scale.view(1, C, 1, 1) * (out - x_feat)
            out = self.proj_out(out, time_emb_in)

            return out

        if self.training:
            return checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            return main_path(x, time_emb)


class Stem(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 factory: Factory, ):
        super().__init__()

        assert in_channels == 3, "in_channels should be 3."

        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False
        )
        self.cln = factory.get_cln(channels=out_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, time_emb)
        out = self.cln(out, time_emb)
        return out


class TimeAwareToRGB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 factory: Factory, ):
        super().__init__()
        assert out_channels == 3, "out_channels should be 3."
        self.cln = factory.get_cln(channels=in_channels)
        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        x = self.cln(x, time_emb)
        return self.conv(x, time_emb)


class DiffusionTransUNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 time_emb_dim=128,
                 config: TransUNetConfig | None = None):
        super().__init__()
        self.config = config or TransUNetConfig(time_emb_dim=time_emb_dim)
        self._validate_config()
        self.time_emb_dim = self.config.time_emb_dim

        c = self.config
        rope_size = c.rope_max_size if c.rope_max_size is not None else c.image_size
        rope_height, rope_width = _to_2tuple(rope_size, "rope_max_size" if c.rope_max_size is not None else "image_size")

        factory = Factory(
            time_emb_dim=c.time_emb_dim,
            convolution_kernel_group=c.num_experts,
            attn_head_dim=c.attn_head_dim,
            rope_height=rope_height,
            rope_width=rope_width,
            rope_base=c.rope_base,
            router_temperature=c.router_temperature,
            collect_router_auxiliary_loss=c.collect_router_auxiliary_loss
        )
        self.time_mlp = factory.get_time_mlp(
            hidden_dim=c.time_hidden_dim,
            num_layers=c.time_layers
        )

        self.stem = Stem(in_channels=in_channels, out_channels=c.stem_channels, factory=factory)

        num_stages = len(c.encoder_channels)
        decoder_channels = c.decoder_channels or tuple(reversed(c.encoder_channels))

        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels: list[int] = []

        current_channels = c.stem_channels
        for level, (res_channels, num_blocks) in enumerate(zip(c.encoder_channels, c.encoder_blocks)):
            attn_heads = c.encoder_attn_heads if level in c.encoder_attn_levels else None
            stage = AttentiveDualPathStage(
                in_channels=current_channels,
                res_channels=res_channels,
                num_blocks=num_blocks,
                dense_inc=c.dense_inc,
                groups=c.groups,
                factory=factory,
                attn_heads=attn_heads
            )
            self.encoder_stages.append(stage)
            current_channels = stage.out_channels
            self.skip_channels.append(current_channels)

            down_out_channels = c.encoder_channels[level + 1] if level < num_stages - 1 else c.bottleneck_channels
            self.downsamples.append(
                DownsampleLayer(current_channels, down_out_channels, dense_inc=0, factory=factory)
            )
            current_channels = down_out_channels

        self.bot_stage = BottleneckTransformerStage(
            in_channels=c.bottleneck_channels,
            out_channels=c.bottleneck_channels,
            inner_channels=c.bottleneck_inner_channels,
            num_layers=c.bottleneck_layers,
            num_heads=c.bottleneck_heads,
            factory=factory
        )

        self.upsamples = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        current_channels = c.bottleneck_channels
        for level, (skip_channels, res_channels, num_blocks) in enumerate(
                zip(reversed(self.skip_channels), decoder_channels, c.decoder_blocks)
        ):
            self.upsamples.append(
                PixelShuffleUpsampleLayer(current_channels, res_channels, dense_inc=0, factory=factory)
            )
            self.fusions.append(
                FeatureFusionBlock(res_channels + skip_channels, res_channels, factory=factory)
            )

            attn_heads = c.decoder_attn_heads if level in c.decoder_attn_levels else None
            stage = AttentiveDualPathStage(
                in_channels=res_channels,
                res_channels=res_channels,
                num_blocks=num_blocks,
                dense_inc=c.dense_inc,
                groups=c.groups,
                factory=factory,
                attn_heads=attn_heads
            )
            self.decoder_stages.append(stage)
            current_channels = stage.out_channels

        self.final = TimeAwareToRGB(in_channels=current_channels, out_channels=out_channels, factory=factory)

        # 在初始化时缓存所有 CondConv 层，避免每次计算 loss 时遍历整个网络
        self.cond_conv_layers = [
            m for m in self.modules() if isinstance(m, TimeAwareCondConv2d)
        ]

    def _validate_config(self):
        c = self.config
        num_stages = len(c.encoder_channels)
        if num_stages == 0:
            raise ValueError("encoder_channels must contain at least one stage.")
        if len(c.encoder_blocks) != num_stages:
            raise ValueError("encoder_blocks must have the same length as encoder_channels.")
        if len(c.decoder_blocks) != num_stages:
            raise ValueError("decoder_blocks must have the same length as encoder_channels.")
        if c.decoder_channels is not None and len(c.decoder_channels) != num_stages:
            raise ValueError("decoder_channels must be None or have the same length as encoder_channels.")
        if c.attn_head_dim <= 0:
            raise ValueError("attn_head_dim must be positive.")
        if c.attn_head_dim % 4 != 0:
            raise ValueError("attn_head_dim must be divisible by 4 for RoPE2D.")
        if c.router_temperature <= 0:
            raise ValueError("router_temperature must be positive.")
        if c.ortho_weight < 0:
            raise ValueError("ortho_weight must be non-negative.")
        if c.router_balance_weight < 0 or c.router_entropy_weight < 0:
            raise ValueError("router auxiliary weights must be non-negative.")

        image_height, image_width = _to_2tuple(c.image_size, "image_size")
        down_factor = 2 ** num_stages
        if image_height % down_factor != 0 or image_width % down_factor != 0:
            raise ValueError(f"image_size must be divisible by {down_factor}, got {(image_height, image_width)}.")

        self._validate_attention_levels(c.encoder_attn_levels, num_stages, "encoder_attn_levels")
        self._validate_attention_levels(c.decoder_attn_levels, num_stages, "decoder_attn_levels")

        decoder_channels = c.decoder_channels or tuple(reversed(c.encoder_channels))
        self._validate_attention_width(
            channels=c.bottleneck_inner_channels,
            num_heads=c.bottleneck_heads,
            head_dim=c.attn_head_dim,
            name="bottleneck attention"
        )

        current_channels = c.stem_channels
        for level, num_blocks in enumerate(c.encoder_blocks):
            attn_channels = current_channels + num_blocks * c.dense_inc
            if level in c.encoder_attn_levels:
                self._validate_attention_width(
                    channels=attn_channels,
                    num_heads=c.encoder_attn_heads,
                    head_dim=c.attn_head_dim,
                    name=f"encoder attention level {level}"
                )
            current_channels = c.encoder_channels[level + 1] if level < num_stages - 1 else c.bottleneck_channels

        for level, (res_channels, num_blocks) in enumerate(zip(decoder_channels, c.decoder_blocks)):
            if level not in c.decoder_attn_levels:
                continue
            self._validate_attention_width(
                channels=res_channels + num_blocks * c.dense_inc,
                num_heads=c.decoder_attn_heads,
                head_dim=c.attn_head_dim,
                name=f"decoder attention level {level}"
            )

        stage_channels = list(c.encoder_channels)
        if c.decoder_channels is not None:
            stage_channels.extend(c.decoder_channels)
        stage_channels.append(c.bottleneck_channels)
        for channels in stage_channels:
            if channels % c.groups != 0:
                raise ValueError(f"stage channel {channels} must be divisible by groups={c.groups}.")
        if c.stem_channels < c.encoder_channels[0]:
            raise ValueError("stem_channels must be >= the first encoder channel.")

    @staticmethod
    def _validate_attention_levels(levels: tuple[int, ...], num_stages: int, name: str):
        invalid_levels = [level for level in levels if level < 0 or level >= num_stages]
        if invalid_levels:
            raise ValueError(f"{name} contains invalid levels {invalid_levels}; valid range is [0, {num_stages - 1}].")

    @staticmethod
    def _validate_attention_width(channels: int, num_heads: int, head_dim: int, name: str):
        if num_heads <= 0:
            raise ValueError(f"{name} num_heads must be positive, got {num_heads}.")
        expected_heads, remainder = divmod(channels, head_dim)
        if remainder != 0:
            raise ValueError(
                f"{name} channels must be divisible by attn_head_dim; "
                f"got channels={channels}, attn_head_dim={head_dim}, remainder={remainder}."
            )
        if num_heads != expected_heads:
            raise ValueError(
                f"{name} expects num_heads * attn_head_dim == channels; "
                f"got {num_heads} * {head_dim} = {num_heads * head_dim}, channels={channels}. "
                f"Set num_heads to {expected_heads} for attn_head_dim={head_dim}."
            )

    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        x = self.stem(x, t_emb)

        skips = []
        for stage, downsample in zip(self.encoder_stages, self.downsamples):
            x = stage(x, t_emb)
            skips.append(x)
            x = downsample(x, t_emb)

        x = self.bot_stage(x, t_emb)

        for upsample, fusion, stage, skip in zip(
                self.upsamples,
                self.fusions,
                self.decoder_stages,
                reversed(skips)
        ):
            x = upsample(x, t_emb)
            x = torch.cat([x, skip], dim=1)
            x = fusion(x, t_emb)
            x = stage(x, t_emb)

        return self.final(x, t_emb)

    def set_router_auxiliary_loss_enabled(self, enabled: bool):
        enabled = bool(enabled)
        self.config.collect_router_auxiliary_loss = enabled
        for module in self.cond_conv_layers:
            module.set_router_auxiliary_loss_enabled(enabled)

    def get_auxiliary_loss(self, ortho_weight=None):
        ortho_weight = self.config.ortho_weight if ortho_weight is None else ortho_weight
        if ortho_weight <= 0:
            return next(self.parameters()).new_zeros(())

        ortho_loss_total = next(self.parameters()).new_zeros(())
        for module in self.cond_conv_layers:
            ortho_loss_total += module.get_ortho_loss()

        return ortho_weight * ortho_loss_total

    def get_router_auxiliary_loss(self, balance_weight=None, entropy_weight=None):
        c = self.config
        if not c.collect_router_auxiliary_loss:
            return next(self.parameters()).new_zeros(())

        balance_weight = c.router_balance_weight if balance_weight is None else balance_weight
        entropy_weight = c.router_entropy_weight if entropy_weight is None else entropy_weight

        if balance_weight <= 0 and entropy_weight <= 0:
            return next(self.parameters()).new_zeros(())

        router_loss_total = next(self.parameters()).new_zeros(())
        for module in self.cond_conv_layers:
            router_loss = module.get_router_loss(
                balance_weight=balance_weight,
                entropy_weight=entropy_weight
            )
            if router_loss is not None:
                router_loss_total = router_loss_total + router_loss

        return router_loss_total


# ================= 测试代码 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 轻量自测使用小图，默认配置仍面向 512x512 训练。
    test_config = TransUNetConfig(image_size=64, rope_max_size=64)
    model = DiffusionTransUNet(config=test_config).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 模拟输入 (Batch Size=1, Channels=3, 64x64)
    image_height, image_width = _to_2tuple(test_config.image_size, "image_size")
    x = torch.randn(1, 3, image_height, image_width).to(device)

    # 模拟时间步 (随机 t)
    test_timesteps = 1000
    t = torch.randint(0, test_timesteps, (x.shape[0],), device=device).float() / test_timesteps

    # 前向传播
    try:
        output = model(x, t)
        print("Input shape:", x.shape)
        print("Output shape:", output.shape)

        assert output.shape == x.shape, "Output shape mismatch!"
        print("\n✅ Test Passed! The UNet is ready for training cat images.")

        # 简单检查是否有 NaN
        if torch.isnan(output).any():
            print("❌ Warning: NaN detected in output.")
        else:
            print("✅ Output numerical check passed.")

    except Exception as e:
        print(f"\n❌ Error during forward pass: {e}")
        import traceback

        traceback.print_exc()