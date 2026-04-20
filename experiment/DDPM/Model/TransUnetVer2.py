import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint


# ================================================
# 2026/4/20 by Xingqian She
# ================================================


class Factory:
    def __init__(self,
                 time_emb_dim,
                 convolution_kernel_group=4,
                 attn_head_dim=64,):
        self.time_emb_dim = time_emb_dim
        self.convolution_kernel_group = convolution_kernel_group
        self.attn_head_dim = attn_head_dim
        self.rope = RoPE2D(dim=self.attn_head_dim)

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
            num_experts=self.convolution_kernel_group
        )

    def get_timemlp(self):
        pass

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

        t_y = torch.arange(height, dtype=torch.float)
        freqs_y = torch.einsum('i,j->ij', t_y, inv_freq)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)


        t_x = torch.arange(width, dtype=torch.float)
        freqs_x = torch.einsum('i,j->ij', t_x, inv_freq)
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)

        emb_y_grid = emb_y.unsqueeze(1).expand(-1, width, -1)

        emb_x_grid = emb_x.unsqueeze(0).expand(height, -1, -1)

        freqs_y_full = emb_y_grid.reshape(-1, half_dim)
        freqs_x_full = emb_x_grid.reshape(-1, half_dim)

        self.register_buffer("cos_y_cached", freqs_y_full.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_y_cached", freqs_y_full.sin().unsqueeze(0).unsqueeze(0))

        self.register_buffer("cos_x_cached", freqs_x_full.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_x_cached", freqs_x_full.sin().unsqueeze(0).unsqueeze(0))

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]

        def main_path(x_in: torch.Tensor):
            y_part, x_part = x_in.chunk(2, dim=-1)

            cos_y = self.cos_y_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)
            sin_y = self.sin_y_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)

            cos_x = self.cos_x_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)
            sin_x = self.sin_x_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)

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


class LoRAHyperLayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super(LoRAHyperLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.a_generator = nn.Sequential(
            nn.Linear(input_dim, max(8, input_dim // 8)),
            nn.SiLU(),
            nn.Linear(max(8, input_dim // 8), input_dim * rank, bias=False)
        )

        self.b_generator = nn.Sequential(
            nn.Linear(input_dim, max(8, input_dim // 8)),
            nn.SiLU(),
            nn.Linear(max(8, input_dim // 8), rank * output_dim, bias=False)
        )

    def forward(self, x)  -> dict[str, torch.Tensor]:

        a_flat = self.a_generator(x)
        b_flat = self.b_generator(x)

        A = a_flat.reshape(-1, self.input_dim, self.rank)
        B = b_flat.reshape(-1, self.rank, self.output_dim)

        return {'A': A, 'B': B}

class HyperContent(nn.Module):
    def __init__(self, input_dim, output_dim, rank = None):
        super().__init__()
        self.rank = rank if rank is not None else max(8, input_dim // 8)
        self.hyper_layer = LoRAHyperLayer(input_dim, output_dim, self.rank)

    def forward(self, x)  -> torch.Tensor:
        params = self.hyper_layer(x)
        A = params['A']
        B = params['B']

        res = torch.einsum('...i, bir, bro -> ...o', x, A, B)
        return res

class HybridSwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, up_proj_dim):
        super().__init__()
        self.static_gate = nn.Linear(input_dim, up_proj_dim)
        self.dynamic_content = HyperContent(input_dim, up_proj_dim, max(8, up_proj_dim // 8))
        self.act = nn.SiLU()
        self.static_down = nn.Linear(up_proj_dim, output_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        static_gate = self.static_gate(x)
        dynamic_content = self.dynamic_content(x)
        res = self.static_down(self.act(static_gate) * dynamic_content)
        return res

class ResMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hybrid_net = HybridSwiGLU(dim,dim,4*dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.hybrid_net(x))


class TimeMLP(nn.Module):
    def __init__(self,
                 time_emb_dim=256,
                 hidden_dim=512,
                 num_layers=6,
                 fourier_scale=16.0):
        super().__init__()
        self.fourier = GaussianFourierProjection(hidden_dim, scale=fourier_scale)
        self.mapping = nn.ModuleList([
            ResMLPBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim, time_emb_dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.dtype == torch.long:
            time = time.float()
        x = self.fourier(time)
        for block in self.mapping:
            x = block(x)
        return self.out_proj(x)


class Modulator(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.hybrid_net = HybridSwiGLU(in_channels, in_channels, 4 * in_channels)

        self.act = nn.SiLU()
        self.to_out = nn.Linear(in_channels, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_out.weight)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        time_emb = self.act(time_emb + self.hybrid_net(time_emb))
        return self.to_out(time_emb)


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
        self.to_q = nn.Linear(time_emb_dim, self.hidden_dim, bias=False)
        self.to_k = nn.Linear(in_channels * 2, self.hidden_dim, bias=False)
        self.to_v = nn.Linear(in_channels * 2, self.hidden_dim, bias=False)
        self.hybrid_net = HybridSwiGLU(hidden_dim, hidden_dim, 4 * hidden_dim)
        self.to_out = nn.Linear(self.hidden_dim, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.to_out.weight)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False) + self.eps
        return torch.cat([mu, std], dim=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        stats = self._compute_stats(x)
        q = self.to_q(time_emb)
        k = self.to_k(stats)
        v = self.to_v(stats)

        attn = torch.sigmoid(q * k)
        x_hidden = v * attn
        x_mixed = x_hidden + self.hybrid_net(x_hidden)
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
                 gating_noise_std: float = 1e-2, ):
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

        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        self.router = CondConvRouter(in_channels, time_emb_dim, num_experts)

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
        n_exp, out_c, in_per_grp, k, k = w.shape

        w_per_channel = w.permute(1, 0, 2, 3, 4).reshape(out_c, n_exp, -1)

        w_norm = F.normalize(w_per_channel, p=2, dim=2)

        gram = torch.bmm(w_norm, w_norm.transpose(1, 2).contiguous())

        identity = torch.eye(n_exp, device=w.device).unsqueeze(0).expand(out_c, n_exp, n_exp)

        diff = torch.norm(gram - identity, p='fro', dim=(1, 2))

        return diff.mean()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        routing_logits = self.router(x, time_emb)
        routing_weights = F.softmax(routing_logits, dim=1)
        weight_eff = (routing_weights.view(B, self.num_experts, 1, 1, 1, 1) *
                      self.weight.unsqueeze(0)).sum(1)

        if self.bias is not None:
            bias_eff = (routing_weights.view(B, self.num_experts, 1) * self.bias.unsqueeze(0)).sum(1)
            bias_eff = bias_eff.view(-1)
        else:
            bias_eff = None
        x_flat = x.view(1, B * C, H, W)
        weight_flat = weight_eff.view(B * self.out_channels,
                                      self.in_channels // self.groups,
                                      self.kernel_size,
                                      self.kernel_size)

        total_groups = B * self.groups

        out_flat = F.conv2d(
            x_flat,
            weight_flat,
            bias_eff,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=total_groups
        )
        return out_flat.view(B, self.out_channels, out_flat.shape[2], out_flat.shape[3])


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        residual = x
        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        x_norm = self.rms(x_flat, time_emb)
        out = self.attn(x_norm)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return residual + out


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

        self.rms2 = factory.get_rms(channels=channels)

        inner_channels = int(channels * ffn_mult)
        self.ffn = factory.get_swiglu(channels=channels, inner_channels=inner_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:

        x = x + self.attn(self.rms1(x, time_emb))
        x = x + self.ffn(self.rms2(x, time_emb), time_emb)
        return x


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

        self.fusion_act = factory.get_act()

        self.fusion_grn = factory.get_grn(inner_channels * 2)

        self.proj_out = factory.get_condconv(
            in_channels=inner_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

        self.trans_scale = nn.Parameter(torch.tensor(1e-2), requires_grad=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor) -> torch.Tensor:
            x_feat = self.proj_in(x_in, time_emb_in)
            B, C, H, W = x_feat.shape
            x_trans = x_feat.flatten(2).transpose(1, 2).contiguous()
            for layer in self.layers:
                x_trans = layer(x_trans, time_emb_in)
            x_trans = x_trans.transpose(1, 2).reshape(B, C, H, W)
            x_trans = x_trans * self.trans_scale
            x_combined = torch.cat([x_feat, x_trans], dim=1)
            x_combined = self.fusion_act(x_combined)
            x_combined = self.fusion_grn(x_combined, time_emb_in)
            out = self.proj_out(x_combined, time_emb_in)

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


class DiffusionTransUNet_64(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = TimeMLP(time_emb_dim=time_emb_dim, hidden_dim=512)

        # 定义工厂类，注意：旋转编码的实现在factory中
        factory = Factory(time_emb_dim=time_emb_dim, convolution_kernel_group=4, attn_head_dim=64)

        # Stem
        self.stem = self.stem = Stem(in_channels=in_channels, out_channels=64, factory=factory)

        # ================= ENCODER =================
        # Enc1: 64ch
        self.enc1 = DualPathStage(in_channels=64, res_channels=64, num_blocks=3, dense_inc=16, groups=4,
                                  factory=factory)
        # Down1: 112 -> 128
        self.down1 = DownsampleLayer(in_channels=112, out_channels=128, dense_inc=0, factory=factory)

        # Enc2: 128ch
        self.enc2 = DualPathStage(in_channels=128, res_channels=128, num_blocks=3, dense_inc=16, groups=4,
                                  factory=factory)
        # Down2: 176 -> 256
        self.down2 = DownsampleLayer(in_channels=176, out_channels=256, dense_inc=0, factory=factory)

        # Enc3: 256ch
        self.enc3 = DualPathStage(in_channels=256, res_channels=256, num_blocks=4, dense_inc=16, groups=4,
                                  factory=factory)
        self.att3 = SpatialSelfAttention(channels=320, num_heads=5, factory=factory)
        # Down3: 320 -> 512
        self.down3 = DownsampleLayer(in_channels=320, out_channels=512, dense_inc=0, factory=factory)

        # ================= BOTTLENECK =================
        self.bot_stage = BottleneckTransformerStage(in_channels=512, out_channels=512, inner_channels=1024,
                                                    num_layers=10,
                                                    num_heads=16, factory=factory)

        # ================= DECODER =================

        # --- Up 1 (8x8 -> 16x16) ---
        # Bot Out (512) -> Up (256)
        self.up1 = PixelShuffleUpsampleLayer(in_channels=512, out_channels=256, dense_inc=0, factory=factory)

        # Concat: Up1(256) + Enc3_Skip(320) = 576
        # Fusion: 560 -> 384
        self.fuse1 = FeatureFusionBlock(in_channels=576, out_channels=384, factory=factory)

        # Stage: 处理 384 通道
        self.dec1 = DualPathStage(in_channels=384, res_channels=384, num_blocks=4, dense_inc=16, groups=4,
                                  factory=factory)
        self.dec1_att = SpatialSelfAttention(channels=448, num_heads=7, factory=factory)

        # --- Up 2 (16x16 -> 32x32) ---
        # Dec1 Out (448) -> Up (192)
        self.up2 = PixelShuffleUpsampleLayer(in_channels=448, out_channels=192, dense_inc=0, factory=factory)

        # Concat: Up2(192) + Enc2_Skip(176) = 368
        # Fusion: 368 -> 192
        self.fuse2 = FeatureFusionBlock(in_channels=368, out_channels=192, factory=factory)

        self.dec2 = DualPathStage(in_channels=192, res_channels=192, num_blocks=3, dense_inc=16, groups=4,
                                  factory=factory)
        # Dec2 Out: 240

        # --- Up 3 (32x32 -> 64x64) ---
        # Dec2 Out (240) -> Up (96)
        self.up3 = PixelShuffleUpsampleLayer(in_channels=240, out_channels=96, dense_inc=0, factory=factory)

        # Concat: Up3(96) + Enc1_Skip(112) = 208
        # Fusion: 208 -> 96
        self.fuse3 = FeatureFusionBlock(in_channels=208, out_channels=96, factory=factory)

        self.dec3 = DualPathStage(in_channels=96, res_channels=96, num_blocks=3, dense_inc=16, groups=4,
                                  factory=factory)
        # Dec3 Out: 144

        # ================= OUTPUT =================
        self.final = TimeAwareToRGB(in_channels=144, out_channels=out_channels, factory=factory)

        # 在初始化时缓存所有 CondConv 层，避免每次计算 loss 时遍历整个网络
        self.cond_conv_layers = [
            m for m in self.modules() if isinstance(m, TimeAwareCondConv2d)
        ]

    def forward(self, x, time):
        t_emb = self.time_mlp(time)
        x = self.stem(x, t_emb)

        # Encoder
        h1 = self.enc1(x, t_emb)
        x = self.down1(h1, t_emb)

        h2 = self.enc2(x, t_emb)
        x = self.down2(h2, t_emb)

        h3 = self.enc3(x, t_emb)
        h3 = self.att3(h3, t_emb)
        x = self.down3(h3, t_emb)

        # Bottleneck
        x = self.bot_stage(x, t_emb)

        # Decoder
        # Layer 1
        x = self.up1(x, t_emb)
        x = torch.cat([x, h3], dim=1)
        x = self.fuse1(x, t_emb)
        x = self.dec1(x, t_emb)
        x = self.dec1_att(x, t_emb)

        # Layer 2
        x = self.up2(x, t_emb)
        x = torch.cat([x, h2], dim=1)
        x = self.fuse2(x, t_emb)
        x = self.dec2(x, t_emb)

        # Layer 3
        x = self.up3(x, t_emb)
        x = torch.cat([x, h1], dim=1)
        x = self.fuse3(x, t_emb)
        x = self.dec3(x, t_emb)

        return self.final(x, t_emb)

    def get_auxiliary_loss(self, ortho_weight=1e-6):
        """
        在计算主 Loss后，调用此函数获取额外的正则化 Loss。
        total_loss = mse_loss + aux_loss
        """
        ortho_loss_total = 0.0
        for module in self.cond_conv_layers:
            ortho_loss_total += module.get_ortho_loss()

        return ortho_weight * ortho_loss_total


# ================= 测试代码 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 1. 实例化模型
    model = DiffusionTransUNet_64().to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f} M")

    # 2. 模拟输入 (Batch Size=4, Channels=3, 64x64)
    x = torch.randn(4, 3, 64, 64).to(device)

    # 3. 模拟时间步 (随机 t)
    t = torch.randint(0, 1000, (4,)).to(device)

    # 4. 前向传播
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