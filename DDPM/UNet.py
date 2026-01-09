import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint

class CondConv2d(nn.Module):
    """
    Conditionally Parameterized Convolution (CondConv).
    Generates input-dependent convolution kernels by mixing multiple expert kernels.
    Supports groups and bias=False as in the original model.

    Reference: https://arxiv.org/pdf/1904.04971
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 bias=False,
                 num_experts=4,
                 gating_noise_std=1e-2,):
        super(CondConv2d, self).__init__()
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

        # Routing network: global avg pool + FC + sigmoid
        self.routing_fc = nn.Linear(in_channels, num_experts)

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
        # temp_weight is the view of self.weight, modifying the view will update the original tensor

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        B, C, H, W = x.shape

        # Compute per-input routing weights
        pooled = F.avg_pool2d(x, (H, W)).view(B, C)  # [B, in_channels]

        # [B, num_experts]
        if self.training:
            gating_logits = self.routing_fc(pooled)
            # In order to alleviate the experts' drowsiness during training and their inability to receive training.
            noise = torch.randn_like(gating_logits) * self.gating_noise_std
            routing_weights = torch.sigmoid(gating_logits + noise)
        else:
            routing_weights = torch.sigmoid(self.routing_fc(pooled))

        # Compute effective weights and biases
        weight_eff = (routing_weights[:, :, None, None, None, None] * self.weight[None]).sum(1)  # [B, out, in//g, k, k]
        if self.bias is not None:
            bias_eff = (routing_weights[:, :, None] * self.bias[None]).sum(1)  # [B, out]
        else:
            bias_eff = None

        # Flatten for batched convolution
        x_flat = x.view(1, B * C, H, W)  # [1, B*in, H, W]
        weight_flat = weight_eff.view(B * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        groups = B * self.groups

        out_flat = F.conv2d(
            x_flat, weight_flat, None, self.stride, self.padding, dilation=1, groups=groups
        )  # [1, B*out, H', W']

        # Add bias if present
        if bias_eff is not None:
            out_flat = out_flat + bias_eff.view(1, B * self.out_channels, 1, 1)

        # Reshape back to [B, out, H', W']
        return out_flat.view(B, self.out_channels, out_flat.shape[2], out_flat.shape[3])


class ChannelLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization as used in ConvNeXt.
    Normalizes across channels for each spatial position.
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization (GRN) as introduced in ConvNeXt V2.
    This normalization encourages competition among channels by amplifying
    stronger responses and suppressing weaker ones, helping to prevent feature collapse.

    Reference: https://arxiv.org/pdf/2301.00808
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Args:
            num_channels (int): Number of channels in the input tensor.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        """
        super(GlobalResponseNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)  # Shape: (B, C, 1, 1)
        x = self.gamma * (x * nx) + self.beta + x
        return x

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
                return CondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.ln = ChannelLayerNorm(inc)
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
        self.grn = GlobalResponseNorm(mid_channels)

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


    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
        Return:
            Tensor
        """
        def main_path(x_in):
            out = self.conv1(self.ln(x_in))
            out = self.conv2(out)
            out = self.conv3_up(out)
            out = self.gelu(out)
            out = self.grn(out)
            # 共享模式
            # out = self.conv3_down(out)
            # return out
            # 分离模式
            out_res = self.conv3_res(out)
            out_dense = self.conv3_dense(out)
            return out_res, out_dense


        if self.training:
            # 共享模式
            # out = checkpoint(main_path, x, use_reentrant=False)
            # 分离模式
            new_res, new_dense = checkpoint(main_path, x, use_reentrant=False)
        else:
            # 共享模式
            # out = main_path(x)
            # 分离模式
            new_res, new_dense = main_path(x)

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
        use_condconv (bool): If True, uses CondConv2d for convolutions.
        num_experts (int): Number of experts for CondConv2d.
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
                 use_condconv=False,
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
                return CondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.ln = ChannelLayerNorm(in_channels)
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
        use_condconv: Whether to use CondConv for convolution.
        num_experts: Number of experts for CondConv.
    """

    def __init__(self,
                 in_channels,
                 out_channels: int,
                 use_condconv: bool = False,
                 num_experts: int = 4):

        super().__init__()

        assert in_channels == 3
        self.out_channels = out_channels
        self.use_condconv = use_condconv
        self.num_experts = num_experts

        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return CondConv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False, num_experts=num_experts)
            else:
                return nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=g, bias=False)

        self.conv = make_conv(in_channels, out_channels, 4, s=4, p=0)
        self.ln = ChannelLayerNorm(out_channels)

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
