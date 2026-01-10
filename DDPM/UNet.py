import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint


class TimeModulator(nn.Module):
    """
    用于 AdaCLN 和 AdaGRN 的时间映射模块
    结构: Linear -> GELU -> Linear
    """

    def __init__(self, time_emb_dim, out_channels, gamma_init_value=0.0):
        super().__init__()
        assert out_channels % 2 == 0, "TimeModulator's out_channels must be even for two part [gamma, beta]"
        self.out_channels = out_channels
        self.gamma_init_value = gamma_init_value  # 保存期望的 gamma 初始值

        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self._init_weights()

    def _init_weights(self):
        # 1. 最后一层的权重 weight 初始化为 0
        # 这样 t (时间) 的变化在初始阶段不会造成剧烈的特征波动
        nn.init.zeros_(self.net[-1].weight)

        # 2. 最后一层的偏置 bias 进行特殊初始化
        # 我们假设 out_channels 是 [gamma, beta] 的拼接
        # 前一半是 gamma，后一半是 beta
        half_dim = self.out_channels // 2

        # 初始化 Gamma 部分 (Bias)
        nn.init.constant_(self.net[-1].bias[:half_dim], self.gamma_init_value)

        # 初始化 Beta 部分 (Bias) 始终为 0
        nn.init.constant_(self.net[-1].bias[half_dim:], 0.0)

    def forward(self, x):
        return self.net(x)


class TimeAttentionRouter(nn.Module):
    def __init__(self, in_channels, time_emb_dim, num_experts=4):
        super().__init__()

        # 1. Q 映射 (Time)
        self.to_q = nn.Linear(time_emb_dim, in_channels)
        # 2. K 映射 (Content)
        self.to_k = nn.Linear(in_channels, in_channels)
        # 3. V 映射 (Content)
        self.to_v = nn.Linear(in_channels, in_channels)

        # 4. 通道混合矩阵 (如果想做通道间交互)
        # 这是一个激进的设计：让通道之间可以相互说话
        # 但参数量是 C*C，如果 C 很大要注意。这里用 reduction 降低风险
        self.channel_mixer = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.GELU(),
            nn.Linear(in_channels // 4, in_channels)
        )

        # 5. 最终分类器
        self.classifier = nn.Linear(in_channels, num_experts)

    def forward(self, x_pooled, time_emb):
        """
        x_pooled: (B, C)
        time_emb: (B, T_dim)
        """
        # 生成 Query (根据时间，我想要什么样的特征？)
        q = self.to_q(time_emb)  # (B, C)

        # 生成 Key (图像实际有什么特征？)
        k = self.to_k(x_pooled)  # (B, C)

        # 计算注意力分数 (Element-wise dot product -> Sigmoid)
        # 这里实际上是计算：时间对每个通道的“关注度”
        attn_weights = torch.sigmoid(q * k)  # (B, C)

        # 应用注意力 (加权)
        v = self.to_v(x_pooled)
        x_attended = v * attn_weights  # (B, C)
        # 此时 x_attended 已经被时间筛选过了：
        # 如果是噪声阶段，attn_weights 可能抑制了高频纹理通道

        # 通道混合 (Channel Mixing)
        # 允许通道之间交换信息 (比如：看到了眼睛，也要激活鼻子的特征)
        x_mixed = x_attended + self.channel_mixer(x_attended)

        # 最终路由
        logits = self.classifier(x_mixed)
        return logits


class TimeAwareCondConv2d(nn.Module):
    """
    Time-Aware CondConv: Dynamically aggregates expert kernels using both input features and time embeddings.
    Allows the convolution to adapt its behavior based on the current diffusion timestep (t).
    Maintains standard Conv2d interface while supporting groups and efficient batch processing.
    """

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
        super(TimeAwareCondConv2d, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts
        self.gating_noise_std = gating_noise_std

        # Expert weights: [num_experts, out_channels, in_channels//groups, k, k]
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        # Routing network: MLP taking both (AvgPooled Input + Time Embedding)
        self.router = TimeAttentionRouter(in_channels, time_emb_dim, num_experts)

        self._init_weights()

    def _init_weights(self):
        # Reshaping weights as [num_experts * out_channels, in_channels // groups, kernel_size, kernel_size]
        temp_weight = self.weight.view(
            self.num_experts * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        # Apply Kaiming normal initialization (applicable to GELU, as it is similar to ReLU)
        init.kaiming_normal_(temp_weight, a=0, mode='fan_in', nonlinearity='relu')

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, time_emb) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature map [B, C, H, W].
            time_emb (Tensor): Time embedding vector [B, D] indicating noise level.
        Returns:
            Tensor: Output feature map computed with time-adaptive kernels.
        """
        B, C, H, W = x.shape

        # Compute per-input routing weights using global context
        pooled = F.avg_pool2d(x, (H, W)).view(B, C)  # [B, in_channels]

        # Calculate routing weights based on Content + Time
        # [B, num_experts]
        if self.training:
            gating_logits = self.router(pooled, time_emb)
            # Add noise during training to encourage expert diversity
            noise = torch.randn_like(gating_logits) * self.gating_noise_std
            routing_weights = torch.tanh(gating_logits + noise)
        else:
            routing_weights = torch.tanh(self.router(pooled, time_emb))

        # Compute effective weights by aggregating experts: Sum(weight_i * routing_i)
        # weight_eff: [B, out, in//g, k, k]
        weight_eff = (routing_weights[:, :, None, None, None, None] * self.weight[None]).sum(1)

        # Compute effective bias if present
        if self.bias is not None:
            bias_eff = (routing_weights[:, :, None] * self.bias[None]).sum(1)  # [B, out]
        else:
            bias_eff = None

        # Flatten batch and channels for grouped convolution trick
        x_flat = x.view(1, B * C, H, W)  # [1, B*in, H, W]
        weight_flat = weight_eff.view(B * self.out_channels, self.in_channels // self.groups, self.kernel_size,
                                      self.kernel_size)
        groups = B * self.groups

        # Apply convolution with unique kernel per sample
        out_flat = F.conv2d(
            x_flat, weight_flat, None, self.stride, self.padding, dilation=1, groups=groups
        )  # [1, B*out, H', W']

        # Add bias if present
        if bias_eff is not None:
            out_flat = out_flat + bias_eff.view(1, B * self.out_channels, 1, 1)

        # Reshape back to standard batch format [B, out, H', W']
        return out_flat.view(B, self.out_channels, out_flat.shape[2], out_flat.shape[3])


class AdaCLN(nn.Module):
    """
    Adaptive Channel Layer Norm (AdaCLN)
    Replaces static weight/bias with time-dependent modulation.
    """

    def __init__(self, num_channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps

        # 将时间嵌入映射为 Scale (gamma) 和 Shift (beta)
        self.emb_layers = TimeModulator(time_emb_dim, 2 * num_channels, 0)

    def forward(self, x, time_emb):
        """
        x: (B, C, H, W)
        time_emb: (B, T_dim)
        """
        # 1. 计算 LayerNorm 的统计量 (Pixel-wise, across channels)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)

        # 2. 获取动态参数
        emb = self.emb_layers(time_emb)  # (B, 2*C)
        gamma, beta = emb.chunk(2, dim=1)  # (B, C), (B, C)

        # 3. 广播并应用 (Modulate)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return x_norm * (1 + gamma) + beta


class AttentiveAdaGRN(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        # 这一部分用来生成 gamma 和 beta
        # 输入不仅仅是 time，而是 time 和 pooled_x 的交互
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.content_proj = nn.Linear(channels, channels)

        # 生成 gamma, beta
        self.to_params = TimeModulator(channels, 2 * channels, 0)

    def forward(self, x, time_emb):
        # x: (B, C, H, W)
        # 1. 标准 GRN 计算 (归一化部分)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)

        # 2. 生成动态参数 (Attention 思想)
        x_pooled = x.mean(dim=(2, 3))  # (B, C)

        # Q-K 交互：时间 Q 调节 内容 K
        # 这里的交互方式：简单相加或相乘，然后过 MLP
        t_vec = self.time_proj(time_emb)
        c_vec = self.content_proj(x_pooled)

        # 融合向量：结合了“现在的噪声水平”和“现在的图像语义”
        context = t_vec * c_vec

        # 映射为参数
        params = self.to_params(context)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # 3. 应用
        return gamma * (x * nx) + beta + x

class DualPathBlock(nn.Module):
    """
    ConvNeXt-like Dual Path Block: Combines residual and dense connections.

    inspired by ConvNeXt V1/V2:
        Larger 7x7 kernel for spatial mixing.
        Inverted bottleneck MLP in pointwise part (1x1 up -> GELU -> GRN -> 1x1 down).
        Minimal Channel-wise LayerNorm (only one at the block start, following ConvNeXt philosophy).
        GELU activation.
        GRN from ConvNeXt V2 to prevent feature collapse.

    Args:
        inc: total input channels (res_channels + dense_channels)
        res_channels: channels for the residual path (fixed per stage)
        dense_inc: additional channels to add to the dense path
        time_emb_dim: the dim from time network's output
        groups: number of groups for grouped convolution
        use_condconv: whether to use CondConv for convolutions
        num_experts: number of experts for CondConv
        expansion_ratio: expansion ratio in ConvNeXt

    Reference:
        https://arxiv.org/pdf/1707.01629
        https://arxiv.org/pdf/2201.03545
        https://arxiv.org/pdf/2301.00808
    """
    def __init__(self,
                 inc,
                 res_channels,
                 dense_inc,
                 time_emb_dim,
                 groups=32,
                 use_condconv=True,
                 num_experts=4,
                 expansion_ratio=4,):
        super(DualPathBlock, self).__init__()
        assert res_channels % groups == 0, f"res_channels:{res_channels} must be divisible by groups:{groups}."

        self.inc = inc
        self.res_channels = res_channels
        self.dense_inc = dense_inc
        self.groups = groups
        self.use_condconv = use_condconv
        self.num_experts = num_experts
        self.expansion_ratio = expansion_ratio

        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.ln = AdaCLN(inc, time_emb_dim)
        self.conv1 = make_conv(inc, res_channels, 1)

        self.conv2 = make_conv(res_channels, res_channels, 7, p=3, g=groups)

        mid_channels = res_channels * expansion_ratio
        self.conv3_up = make_conv(res_channels, mid_channels, 1)

        # LayerScale 参数
        # gamma_res 对应 new_res 的通道数 (res_channels)
        # 初始化为一个非常小的数 (epsilon)，例如 1e-6
        # 这样初始时，new_res * gamma_res ≈ 0，相当于恒等映射
        self.gamma_res = nn.Parameter(1e-6 * torch.ones(res_channels), requires_grad=True)

        # 对于 Dense 部分，如果也想让它初始“隐身”，也可以加一个
        # 作用：保证初期 concatenation 的部分接近 0，不破坏下一层的输入分布
        self.gamma_dense = nn.Parameter(1e-6 * torch.ones(dense_inc), requires_grad=True)


        self.gelu = nn.GELU()

        # GRN
        self.grn = AttentiveAdaGRN(mid_channels, time_emb_dim)

        """
        在 Dual Path Block 中，
        Dense Path（需要拼接传给下一层的特征）和 Residual Path（用于当前层加和的特征），
        应该共享同一个 CondConv 生成的卷积核，还是应该分开处理？
        1. 特征耦合性
            共享模式:
                强制耦合：
                    同一个 CondConv 的专家（Experts）必须同时负责生成 Residual 特征和 Dense 特征。
                    这意味着，路由网络（Router）选择某个专家时，该专家必须“全能”。
                潜在冲突：
                    Residual 特征倾向于“修正”（比如去除噪声、微调纹理），
                    而 Dense 特征倾向于“保存”（保留高频信息、边缘、原始语义供后续使用）。
                    让同一个卷积核同时做这两件事，可能会产生目标冲突。
            分离模式:
                解耦：
                    Residual 分支可以专注于当前层的特征提纯，Dense 分支可以专注于挖掘新颖的特征供深层网络使用。
                灵活性：
                    对于某些样本，可能需要很强的 Residual 修正，但不需要太多的新特征；反之亦然。
                    分离后，两个 Router 可以根据输入做出不同的选择。
        2. 专家专业度
            共享模式：
                专家必须是“通才”。
            分离模式：
                Res 专家：可能专门学习如何做残差去噪、平滑。
                Dense 专家：可能专门学习如何提取边缘、特定形状、特定语义。
        3. 参数效率与计算成本
            共享模式：
                省参数，省路由计算开销。
            分离模式：
                需要两个 Router，做两次权重组合。
        """
        # 共享模式
        # self.conv3_down = make_conv(mid_channels, res_channels + dense_inc, 1)

        # 分离模式
        self.conv3_res = make_conv(mid_channels, res_channels, 1)
        self.conv3_dense = make_conv(mid_channels, dense_inc, 1)


    def forward(self, x, time_emb) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        def main_path(x_in, time_emb):
            out = self.conv1(self.ln(x_in, time_emb), time_emb)
            out = self.conv2(out, time_emb)
            out = self.conv3_up(out, time_emb)
            out = self.gelu(out)
            out = self.grn(out, time_emb)
            # 共享模式
            # out = self.conv3_down(out)
            # return out
            # 分离模式
            out_res = self.conv3_res(out, time_emb)
            out_dense = self.conv3_dense(out, time_emb)
            return out_res, out_dense


        if self.training:
            # 共享模式
            # out = checkpoint(main_path, x, use_reentrant=False)
            # 分离模式
            new_res, new_dense = checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            # 共享模式
            # out = main_path(x)
            # 分离模式
            new_res, new_dense = main_path(x, time_emb)

        # 共享模式
        # new_res = out[:, :self.res_channels, :, :]
        # new_dense = out[:, self.res_channels:, :, :]
        # 分离模式 不需要此操作

        prev_res = x[:, :self.res_channels, :, :]
        prev_dense = x[:, self.res_channels:, :, :]
        # 应用 LayerScale
        # 将 new_res 乘上可学习的 gamma_res
        # gamma_res 维度是 [C]，通过广播变为 [1, C, 1, 1]
        residual = prev_res + new_res * self.gamma_res.view(1, -1, 1, 1)
        if prev_dense.size(1) > 0:
            dense = torch.cat((prev_dense, new_dense * self.gamma_dense.view(1, -1, 1, 1)), dim=1)
        else:
            dense = new_dense * self.gamma_dense.view(1, -1, 1, 1)

        return torch.cat((residual, dense), dim=1)


class DualPathStage(nn.Module):
    """
    A stage of DualPathBlocks.

    This module sequentially applies a specified number of DualPathBlocks.
    Its primary responsibility is to deepen the network representation at a
    constant spatial resolution.

    Args:
        in_channels (int): Total input channels (res_channels + initial_dense).
        res_channels (int): Number of channels for the residual path in this stage.
        num_blocks (int): Number of DualPathBlocks in this stage.
        dense_inc (int): Number of dense channels added by each block.
        groups (int): Number of groups for grouped convolutions.
        use_condconv (bool): If True, uses TimeAwareCondConv2d for convolutions.
        num_experts (int): Number of experts for TimeAwareCondConv2d.
        expansion_ratio (int): Expansion ratio for the inverted bottleneck.
    """

    def __init__(self,
                 in_channels: int,
                 res_channels: int,
                 num_blocks: int,
                 dense_inc: int,
                 groups: int,
                 use_condconv: bool = True,
                 num_experts: int = 4,
                 expansion_ratio: int = 4):
        super().__init__()

        # Calculate initial dense channels from input and residual channels
        initial_dense = in_channels - res_channels
        assert initial_dense >= 0, "res_channels cannot be greater than in_channels."

        current_channels = in_channels
        blocks = []
        for _ in range(num_blocks):
            block = DualPathBlock(
                inc=current_channels,
                res_channels=res_channels,
                dense_inc=dense_inc,
                groups=groups,
                use_condconv=use_condconv,
                num_experts=num_experts,
                expansion_ratio=expansion_ratio
            )
            blocks.append(block)
            current_channels += dense_inc

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        return self.blocks(x)


class DownsampleLayer(nn.Module):
    """
    DownsampleLayer for DPN: Downsamples feature maps and adjusts channel dimensions.
    ConvNeXt-like: LN + 2x2 strided conv.

    Args:
        in_channels: Input channels (res_channels + dense_channels).
        out_channels: Output channels for residual path in next stage.
        dense_inc: Initial dense channels for next stage.
        groups: number of groups for grouped convolution.
        use_condconv: Whether to use CondConv for convolution.
        num_experts: Number of experts for CondConv.


    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dense_inc,
                 groups=1,
                 use_condconv=True,
                 num_experts=4,):
        super(DownsampleLayer, self).__init__()
        assert in_channels % groups == 0, f"in_channels:{in_channels} must be divisible by groups:{groups}."
        assert (out_channels + dense_inc) % groups == 0, f"(out_channels:{out_channels} + dense_inc:{dense_inc}) must be divisible by groups:{groups}."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_inc = dense_inc
        self.use_condconv = use_condconv
        self.num_experts = num_experts

        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.ln = AdaCLN(in_channels)
        self.conv = make_conv(in_channels, out_channels + dense_inc, 2, s=2, p=0, g=groups)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        def main_path(x_in):
            return self.conv(self.ln(x_in))

        if self.training:
            return checkpoint(main_path, x, use_reentrant=False)
        else:
            return main_path(x)

class Stem(nn.Module):
    """
    Stem for DPN: ConvNeXt-like: 4x4 strided conv + LN.

    Args:
        in_channels: Input channels
        out_channels: Output channels after stem (matches first stage res_channels).
        kernel_size: default 4 for classification
        stride: default 4 for classification
        padding: default 0 for classification
        use_condconv: Whether to use CondConv for convolution.
        num_experts: Number of experts for CondConv.
    """

    def __init__(self,
                 in_channels,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 4,
                 padding:int = 0,
                 use_condconv: bool = False,
                 num_experts: int = 4):

        super().__init__()

        assert in_channels == 3
        self.out_channels = out_channels
        self.use_condconv = use_condconv
        self.num_experts = num_experts

        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.conv = make_conv(in_channels, out_channels, kernel_size, s=stride, p=padding)
        self.ln = AdaCLN(out_channels)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        def stem_path(x_in):
            out = self.conv(x_in)
            out = self.ln(out)
            return out

        if self.training:
            return checkpoint(stem_path, x, use_reentrant=False)
        else:
            return stem_path(x)
