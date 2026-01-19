import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint


"""
# SinusoidalPositionalEmbeddings暂时不启用，因为有GaussianFourierProjection了
class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
"""


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier Embeddings (借鉴自 Score-based Generative Modeling)
    比标准的 Sinusoidal 更能捕捉高频变化，适合精细控制。
    """
    def __init__(self, embed_dim, scale=16.0):
        super().__init__()
        # 创建 tensor
        W = torch.randn(embed_dim // 2) * scale

        # 注册为 buffer
        self.register_buffer("W", W)

    def forward(self, x):
        # x: (B,) -> time steps
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class RoPE2D(nn.Module):
    """
    2D 旋转位置编码 (Rotary Position Embedding)
    将 head_dim 的前半部分用于编码 Y(高度) 位置，后半部分用于编码 X(宽度) 位置。

    Args:
        dim: head dimension，必须是偶数
        height: 图像/特征图的高度（patch 数量）
        width: 图像/特征图的宽度（patch 数量）
        base: RoPE 的基础频率 这里我们选用100，因为处理的范围是在8-16，然后在注意力计算中，这个是cos（竖直对）+ cos（水平对）的累加和，
        类似曼哈顿距离，这种情况下，100 Max Wavelength大约628 像素，既没有混叠（远大于16），又有足够的区分度。
    """

    def __init__(
            self,
            dim: int,
            height: int,
            width: int,
            base: float = 100.0
    ):
        super().__init__()

        assert dim % 4 == 0, f"dim must be 4*n, got 4*{dim // 4}+{dim % 4}"

        self.dim = dim
        self.height = height
        self.width = width
        self.base = base

        half_dim = dim // 2  # 每个轴分配的维度
        quarter_dim = half_dim // 2  # 用于生成频率的维度

        # 1. 生成逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, quarter_dim, dtype=torch.float) / quarter_dim))

        # 2. 生成 Height (Y) 的位置编码
        t_y = torch.arange(height, dtype=torch.float)
        freqs_y = torch.einsum('i,j->ij', t_y, inv_freq)  # (H, quarter_dim)
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)  # (H, half_dim)

        # 3. 生成 Width (X) 的位置编码
        t_x = torch.arange(width, dtype=torch.float)
        freqs_x = torch.einsum('i,j->ij', t_x, inv_freq)  # (W, quarter_dim)
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)  # (W, half_dim)

        # 4. 广播构建全图 Grid
        # Y轴编码: (H, 1, half_dim) -> (H, W, half_dim)
        emb_y_grid = emb_y.unsqueeze(1).expand(-1, width, -1)

        # X轴编码: (1, W, half_dim) -> (H, W, half_dim)
        emb_x_grid = emb_x.unsqueeze(0).expand(height, -1, -1)

        # 5. 拼接 -> (H, W, dim)
        emb_full = torch.cat([emb_y_grid, emb_x_grid], dim=-1)

        # 拉平为序列形式 -> (H*W, dim)
        freqs = emb_full.reshape(-1, dim)

        # 6. 预计算并注册 cos/sin 缓存
        # shape: (1, 1, H*W, dim) 方便广播
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0))

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            seq_len: int = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 query 和 key 应用 2D 旋转位置编码

        Args:
            q: Query tensor, shape (B, num_heads, SeqLen, head_dim)
            k: Key tensor, shape (B, num_heads, SeqLen, head_dim)
            seq_len: 可选，实际序列长度（用于处理变长序列）

        Returns:
            tuple: 应用位置编码后的 (q, k)
        """
        if seq_len is None:
            seq_len = q.shape[2]

        # 获取对应长度的 cos/sin
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = self.apply_rotary_pos_emb(q, cos, sin)
        k_embed = self.apply_rotary_pos_emb(k, cos, sin)

        return q_embed, k_embed

    def apply_rotary_pos_emb(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor
    ) -> torch.Tensor:
        """
        应用旋转位置编码
        x: (B, H, SeqLen, Dim)
        cos, sin: (1, 1, SeqLen, Dim)
        """
        return (x * cos) + (self.rotate_half(x) * sin)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将张量的后半部分旋转到前面，并取负"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class ResMLPBlock(nn.Module):
    """
    带残差连接的 MLP Block，允许网络做得更深而不梯度消失
    """
    def __init__(self,
                 dim,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TimeMLP(nn.Module):
    """
    将时间 t 映射为高质量的控制向量。
    """

    def __init__(self,
                 time_emb_dim=256,
                 hidden_dim=512,
                 num_layers=4,
                 fourier_scale=16.0):  # 频率缩放系数
        super().__init__()

        # 1. 高斯傅里叶投影 (Time -> Hidden)
        self.fourier = GaussianFourierProjection(hidden_dim, scale=fourier_scale)

        # 2. 初始映射
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # 3. 深层残差网络 (Deep Mapping)
        # 用来解耦时间步，让 t=500 和 t=501 产生足够不同且平滑变化的特征
        self.mapping = nn.ModuleList([
            ResMLPBlock(hidden_dim) for _ in range(num_layers)
        ])

        # 4. 输出层 (Hidden -> Time Emb Dim)
        self.out_proj = nn.Linear(hidden_dim, time_emb_dim)

    def forward(self, time):
        """
        time: (B,) 输入的时间步索引或归一化时间
        """
        if time.dtype == torch.long:
            time = time.float()

        # 1. 投影
        x = self.fourier(time)

        # 2. 初始激活
        x = self.input_proj(x)

        # 3. 深层处理
        for block in self.mapping:
            x = block(x)

        # 4. 调整维度输出
        return self.out_proj(x)

class TimeModulator(nn.Module):
    """
    用于 AdaCLN 和 AdaGRN 的时间映射模块
    结构: Linear -> SiLU -> Linear
    """

    def __init__(self, time_emb_dim, out_channels, init_value=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.init_value = init_value

        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self._init_weights()

    def _init_weights(self):
        # 1. 最后一层的权重 weight 初始化为 0
        # 这样 t (时间) 的变化在初始阶段不会造成剧烈的特征波动
        nn.init.zeros_(self.net[-1].weight)

        # 2. 最后一层的偏置 bias 进行特殊初始化
        nn.init.constant_(self.net[-1].bias, self.init_value)

    def forward(self, x):
        return self.net(x)


class TimeAttentionRouter(nn.Module):
    def __init__(self, in_channels, time_emb_dim, num_experts=4, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1. Q 映射 (Time) -> 压缩到 hidden_dim
        self.to_q = nn.Linear(time_emb_dim, self.hidden_dim)

        # 2. K 映射 (Content) -> 压缩到 hidden_dim
        self.to_k = nn.Linear(in_channels, self.hidden_dim)

        # 3. V 映射 (Content) -> 压缩到 hidden_dim
        self.to_v = nn.Linear(in_channels, self.hidden_dim)

        # 4. 通道混合矩阵 (Latent Mixer)
        # 在 64 维的隐空间里做交互，参数量(64*64*2)
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 5. 最终分类器
        self.classifier = nn.Linear(self.hidden_dim, num_experts)

    def forward(self, x_pooled, time_emb):
        """
        x_pooled: (B, C) - 全局平均池化后的特征
        time_emb: (B, T_dim)
        """
        # 生成 Query (时间条件的隐特征)
        q = self.to_q(time_emb)  # (B, hidden_dim)

        # 生成 Key (图像内容的隐特征)
        k = self.to_k(x_pooled)  # (B, hidden_dim)

        # 计算注意力/门控分数
        # 这步操作的物理含义：时间 q 决定了它想“激活” latent space 中的哪些特征维度
        attn_weights = torch.sigmoid(q * k)  # (B, hidden_dim)

        # 生成 Value
        v = self.to_v(x_pooled)  # (B, hidden_dim)

        # 应用注意力
        # 筛选出当前时间步下，图像内容中最重要的隐特征
        x_attended = v * attn_weights  # (B, hidden_dim)

        # 混合 (Mixing)
        # 让隐特征之间进行推理，得出最终结论
        x_mixed = x_attended + self.channel_mixer(x_attended)

        # 最终路由
        logits = self.classifier(x_mixed)  # (B, num_experts)
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
                 gating_noise_std: float = 1e-2,):
        super(TimeAwareCondConv2d, self).__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
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
        # 1. 计算 Fan_in
        # 对于单个 Expert，它的感受野包含多少个输入参数
        fan_in = (self.in_channels // self.groups) * (self.kernel_size ** 2)

        # 2. 计算 Kaiming Normal 所需的标准差
        # 2.0 是针对 ReLU/GELU 类激活函数的增益系数 (Gain squared)
        std = math.sqrt(2.0 / fan_in)

        # 3. 直接对 5 维权重进行初始化
        # 将每个专家 (Expert) 都初始化为一个独立的、合格的卷积核
        nn.init.normal_(self.weight, mean=0.0, std=std)

        # 4. 偏置 (Bias) 初始化
        # 偏置通常初始化为 0，或者基于 fan_in 的均匀分布
        if self.bias is not None:
            # 这里的 fan_in 其实应该用 fan_in_bias，通常等于 fan_in
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

        """
        # 这个部分过去是为了图像检测任务设计，在DDPM中，可能加噪声不合适。
        # 大概就是我发现模型loss卡在 0.025左右下不去了（我没有继续训练，可能这个不是下限）
        # 怎么说呢，加了噪声后，模型的输出变成了 N(mu，sigma_{noise}^2)，对于MSE的期望 即:
        # E((hat{y} - y_{true})^2) = (mu - y_{true})^2 + sigma_{noise}^2
        # 这里的这个sigma_{noise}^2是模型永远无法消除的。所以loss会遇到一个无法突破的瓶颈。
        if self.training:
            gating_logits = self.router(pooled, time_emb)
            # Add noise during training to encourage expert diversity
            noise = torch.randn_like(gating_logits) * self.gating_noise_std
            routing_weights = torch.sigmoid(gating_logits + noise)
        else:
            routing_weights = torch.sigmoid(self.router(pooled, time_emb))
        """

        """
        1. 关于噪声：不添加任何人为噪音可能是适合DDPM的。

        2. 关于Softmax vs Sigmoid：
           用softmax可能更适合DDPM。原本的sigmoid激活在图像分类/检测任务中可能影响不大
           （例如当出现某种模式时，就是那个类别，检测任务中的定位的精确度也不像DDPM这么严格），
           但在DDPM这一类要求精确的回归任务中，如果方差不可控，模型可能会消耗很多精力去适应sigmoid组合专家带来的统计分布变化。
           这会导致严重的“内耗”。

        3. 关于正则化：
           目前决定不启用正则化强制让专家负载均衡。
           原因：卷积核不像大语言模型中的专家（MLP），卷积核的参数容量很小，主要是做模式匹配。
           一个核如果被迫去学多种模式，必然导致精度下降。
           在DDPM中，初期步（构图）和后期步（去噪）的任务差距很大，不同的卷积核天然倾向于负责不同阶段的任务。
           如果在某些时刻路由学会了组合专家（例如 0.5*A + 0.5*B），那是为了增强表达能力。
           如果强行加正则，强制保证每一个卷积核的利用率，反而会破坏这种自然的分工，造成模型性能下降。
        """

        # 使用 Softmax 保证权重之和为 1，维持特征图的方差稳定性
        routing_weights = F.softmax(self.router(pooled, time_emb), dim=1)

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
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
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
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

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
        Minimal Channel-wise LayerNorm.
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
                 inc: int,
                 res_channels: int,
                 dense_inc: int,
                 time_emb_dim: int,
                 groups: int=8,
                 use_condconv: bool=True,
                 num_experts: int=4,
                 expansion_ratio: int=4,):
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
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.conv1 = make_conv(inc, res_channels, 1)

        self.conv2 = make_conv(res_channels, res_channels, 7, p=3, g=groups)
        self.ln = AdaCLN(res_channels, time_emb_dim)

        mid_channels = res_channels * expansion_ratio
        # 必须关闭bias（这里bias是默认关闭的），因为bias会影响接下来的GRN工作
        self.conv3_up = make_conv(res_channels, mid_channels, 1)

        # LayerScale 参数
        # gamma_res 对应 new_res 的通道数 (res_channels)
        # 初始化为一个非常小的数 (epsilon)，例如 1e-6
        # 这样初始时，new_res * gamma_res ≈ 0，相当于恒等映射
        self.gamma_res = nn.Parameter(1e-6 * torch.ones(res_channels), requires_grad=True)

        # 对于 Dense 部分，小一些是为了稳定训练，小一些，dense连接对后层的方差影响比较小
        # 需要比gamma_res大，以防止Dense路径失活
        self.gamma_dense = nn.Parameter(1e-2 * torch.ones(dense_inc), requires_grad=True)


        self.gelu = nn.GELU()

        # GRN
        self.grn = AttentiveAdaGRN(mid_channels, time_emb_dim)

        self.conv3_res = make_conv(mid_channels, res_channels, 1)
        self.conv3_dense = make_conv(mid_channels, dense_inc, 1)


    def forward(self, x, time_emb) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        def main_path(x_in: torch.Tensor, time_emb: torch.Tensor):
            out = self.conv1(x_in, time_emb)
            out = self.conv2(out, time_emb)
            out = self.ln(out, time_emb)
            out = self.conv3_up(out, time_emb)
            out = self.gelu(out)
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
        time_emb_dim(int): the dim from time network's output.
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
                 time_emb_dim: int,
                 use_condconv: bool = True,
                 num_experts: int = 4,
                 expansion_ratio: int = 4):
        super().__init__()

        initial_dense = in_channels - res_channels
        assert initial_dense >= 0, "res_channels cannot be greater than in_channels."

        current_channels = in_channels

        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            block = DualPathBlock(
                inc=current_channels,
                res_channels=res_channels,
                dense_inc=dense_inc,
                time_emb_dim=time_emb_dim,
                groups=groups,
                use_condconv=use_condconv,
                num_experts=num_experts,
                expansion_ratio=expansion_ratio
            )
            self.blocks.append(block)
            current_channels += dense_inc

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):

            for block in self.blocks:
                x_in = block(x_in, time_emb_in)
            return x_in

        if self.training:
            return checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            return main_path(x, time_emb)



class FeatureFusionBlock(nn.Module):
    """
    专门用于 Decoder 阶段 skip connection 拼接后的特征融合与降维。
    结构: AdaCLN -> GELU -> 1x1 TimeAwareCondConv
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_emb_dim,
                 use_condconv=True,
                 num_experts=4):
        super().__init__()
        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.act = nn.GELU()
        self.norm = AdaCLN(in_channels, time_emb_dim)
        # 1x1 卷积用于降维，不改变空间尺寸
        self.conv = make_conv(in_channels, out_channels, 1,1,0,1)

    def forward(self, x, time_emb):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            x_in = self.norm(x_in, time_emb_in)
            x_in = self.act(x_in)
            return self.conv(x_in, time_emb_in)

        return main_path(x, time_emb)


class DownsampleLayer(nn.Module):
    """
    DownsampleLayer for DPN: Downsamples feature maps and adjusts channel dimensions.
    ConvNeXt-like: LN + 2x2 strided conv.

    Args:
        in_channels: Input channels (res_channels + dense_channels).
        out_channels: Output channels for residual path in next stage.
        dense_inc: Initial dense channels for next stage.
        time_emb_dim(int): the dim from time network's output.
        groups: number of groups for grouped convolution.
        use_condconv: Whether to use CondConv for convolution.
        num_experts: Number of experts for CondConv.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 time_emb_dim:int,
                 groups: int=1,
                 use_condconv: bool=True,
                 num_experts:int=4,):
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
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.ln = AdaCLN(in_channels, time_emb_dim)
        self.conv = make_conv(in_channels, out_channels + dense_inc, 2, s=2, p=0, g=groups)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            x_in = self.ln(x_in, time_emb_in)
            return self.conv(x_in, time_emb_in)

        return main_path(x, time_emb)


"""
class UpsampleLayer(nn.Module):

    上采样层：Nearest Interpolation + (TimeAwareCondConv or TimeAwareConv)

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 time_emb_dim: int,
                 groups: int=1,
                 use_condconv: bool=True,
                 num_experts: int=4):
        super().__init__()
        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.ln = AdaCLN(in_channels, time_emb_dim)
        self.act = nn.GELU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = make_conv(in_channels, out_channels + dense_inc, 3,1,1,groups)

    def forward(self, x, time_emb):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            x_in = self.ln(x_in, time_emb_in)
            x_in = self.act(x_in)
            x_in = self.upsample(x_in)
            x_in = self.conv(x_in, time_emb_in)
            return x_in

        return main_path(x, time_emb)
"""


class PixelShuffleUpsampleLayer(nn.Module):
    """
    追求质量的 PixelShuffle 上采样层。
    结构: LN -> AdaCLN -> Conv(in, out*4) -> PixelShuffle -> Act

    以下是对这个模块中的一些超参数设置的解释：
        TimeAwareCondConv2d 的 bias=False：
            卷积层的 Bias 是加在通道上的。
            假设输出通道是 4 个，Bias向量为[b_0, b_1, b_2, b_3]
            如果 b_0 很亮，而 b_1 很暗。
            那么无论输入图像是什么，输出图像的所有2x2像素块里，左上角永远比右上角亮。
            这相当于在整张图上叠加了一层永久去不掉的网格纹理。
        为什么要加激活函数呢？
            不加 Act 的含义：这层仅仅是在做“调整分辨率”这件事，不负责非线性的特征变换。
            加 Act呢，这是我的直觉，我觉得通道数是足够传递信息的（应该还冗余），不用担心激活损坏信息，
            同时受到MaxPooling的启发，也许加一个激活，保留下强的信号，过滤弱的信号，可以有利于模型学习更鲁棒的特征
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 time_emb_dim: int,
                 groups: int = 1,
                 use_condconv: bool = True,
                 num_experts: int = 4):
        super().__init__()

        # 目标输出通道数
        target_out = out_channels + dense_inc
        # PixelShuffle 需要 r^2 倍的通道数，这里 scale=2，所以是 4 倍
        expansion = 4

        # 内部卷积输出通道数
        inner_out = target_out * expansion

        # 辅助函数：创建卷积
        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.ln = AdaCLN(in_channels, time_emb_dim)

        # 这里我们使用 3x3 卷积来生成扩展通道，增加感受野
        self.conv = make_conv(in_channels, inner_out, k=3, s=1, p=1, g=groups)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        # 这里的激活函数放在 PixelShuffle 之后
        self.act = nn.GELU()

        # ICNR 初始化
        # 这是一个非常关键的步骤，防止 PixelShuffle 初始化时的棋盘格效应
        if use_condconv:
            # 对于 CondConv，我们对每个 expert 进行 ICNR 初始化
            for i in range(num_experts):
                self._icnr_init(self.conv.weight[i], scale=2)
        else:
            raise TypeError("This feature is not yet implemented.")

    def _icnr_init(self, tensor, scale=2):
        """
        ICNR initialization
        核心逻辑：让 PixelShuffle 后的初始效果等同于 Nearest Neighbor 插值。
        意味着输出的 4 个子像素（对于 scale=2）必须拥有完全相同的权重。
        """
        # tensor shape: [out_c, in_c, k, k]
        # out_c 应该是 new_out_c * (scale ** 2)
        out_c, in_c, k, k = tensor.shape

        assert out_c % (scale ** 2) == 0, f"out_c % (scale ** 2) should be 0, now is {out_c % (scale ** 2)}"

        new_out_c = out_c // (scale ** 2)

        # 1. 生成小的卷积核
        # shape: (new_out_c, in_c, k, k)
        kernel = torch.randn(new_out_c, in_c, k, k) * 0.1

        # 2. 使用 repeat_interleave
        # PixelShuffle 的通道排列是 [r^2, r^2, ...] 这样的块状
        # 所以我们需要 [w0, w0, w0, w0, w1, w1, w1, w1, ...]
        # dim=0 是输出通道维度
        kernel = kernel.repeat_interleave(scale ** 2, dim=0)

        # 此时 kernel shape 变回了 (out_c, in_c, k, k)

        # 3. 赋值
        with torch.no_grad():
            tensor.copy_(kernel)

    def forward(self, x, time_emb):
        def main_path(x_in, time_emb_in):
            x_in = self.ln(x_in, time_emb_in)
            # 先卷积增加通道 (低分辨率)
            x_in = self.conv(x_in, time_emb_in)
            # 再 Shuffle 变大尺寸
            x_in = self.pixel_shuffle(x_in)
            # 最后激活
            x_in = self.act(x_in)
            return x_in

        return main_path(x, time_emb)


class TimeAwareSelfAttention(nn.Module):
    """
    带 AdaCLN 调制的多头自注意力机制 (MHSA)。
    对于 256x256 的图像，建议只在低分辨率层 (32x32, 16x16, 8x8) 使用，
    否则显存会爆炸。
    使用 PyTorch 2.0+ 的 F.scaled_dot_product_attention 进行加速。
    """

    def __init__(self,
                 channels: int,
                 time_emb_dim: int,
                 resolution: tuple | int,
                 num_heads: int=4,
                 head_dim: int=64,
                 use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = num_heads * head_dim

        # AdaCLN，这是 TimeAware 的核心
        self.ln = AdaCLN(channels, time_emb_dim)

        # QKV 生成 (Bias通常设为False以获得纯粹的投影，看个人喜好)
        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)

        # 输出投影
        self.to_out = nn.Linear(inner_dim, channels)

        self.use_rope = use_rope
        if use_rope:
            if isinstance(resolution, int):
                self.resolution = (resolution, resolution)
            else:
                self.resolution = resolution
            # 初始化 RoPE
            # 这里的 dim 必须对应 head_dim，而不是 channels
            self.rope = RoPE2D(
                dim=head_dim,
                height=self.resolution[0],
                width=self.resolution[1]
            )

    def forward(self, x, time_emb):
        """
        x: (B, C, H, W)
        time_emb: (B, t_dim)
        """
        B, C, H, W = x.shape
        N = H * W
        assert (H, W) == self.resolution, f"Input resolution ({H},{W}) does not match initialized resolution {self.resolution}"

        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            # 1. AdaCLN Norm & Reshape (Image -> Sequence)
            # (B, C, H, W) -> (B, N, C)
            norm_x = self.ln(x_in, time_emb_in)
            norm_x = norm_x.permute(0, 2, 3, 1).view(B, N, C)

            # 2. 生成 Q, K, V
            # shape: (B, N, 3 * inner_dim)
            qkv = self.to_qkv(norm_x)

            # 拆分 Q, K, V
            # 此时形状: (B, N, num_heads * head_dim)
            q, k, v = qkv.chunk(3, dim=-1)

            # 变换形状以适配 scaled_dot_product_attention
            # 需要变换为 (B, Num_Heads, Seq_Len, Head_Dim)
            q = q.view(B, N, self.num_heads, -1).transpose(1, 2)
            k = k.view(B, N, self.num_heads, -1).transpose(1, 2)
            v = v.view(B, N, self.num_heads, -1).transpose(1, 2)

            if self.use_rope:
                # 应用 RoPE
                # RoPE 期望输入: (B, H, SeqLen, HeadDim)
                # 输出形状保持不变
                q, k = self.rope(q, k)

            # 3. Flash Attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )

            # 4. 还原形状
            # (B, Num_Heads, N, Head_Dim) -> (B, N, Num_Heads, Head_Dim) -> (B, N, Inner_Dim)
            # 注意：这里必须用 contiguous()，因为 transpose 改变了内存布局
            out = out.transpose(1, 2).contiguous().view(B, N, -1)

            # 5. 输出投影
            out = self.to_out(out)

            # 6. 变回图像空间 (Sequence -> Image)
            # (B, N, C) -> (B, C, H, W)
            out = out.view(B, H, W, C).permute(0, 3, 1, 2)

            # Residual connection
            return x_in + out

        return main_path(x, time_emb)


class Stem(nn.Module):
    """
    Stem for DPN: ConvNeXt-like: 4x4 strided conv + LN.

    Args:
        in_channels: Input channels
        out_channels: Output channels after stem (matches first stage res_channels).
        time_emb_dim(int): the dim from time network's output.
        kernel_size: default 4 for classification
        stride: default 4 for classification
        padding: default 0 for classification
        groups: default 1 for classification
        use_condconv: Whether to use CondConv for convolution.
        num_experts: Number of experts for CondConv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 time_emb_dim: int,
                 kernel_size: int = 4,
                 stride: int = 4,
                 padding:int = 0,
                 groups: int = 1,
                 use_condconv: bool = True,
                 num_experts: int = 4):

        super().__init__()

        assert in_channels == 3
        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.conv = make_conv(in_channels, out_channels, kernel_size, s=stride, p=padding, g=groups)
        self.ln = AdaCLN(out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        def main_path(x_in, time_emb_in):
            out = self.conv(x_in, time_emb_in)
            out = self.ln(out, time_emb_in)
            return out

        return main_path(x, time_emb)


class TimeAwareToRGB(nn.Module):
    """
    将特征映射回图像空间 (RGB或噪声)。
    结构: Norm -> Conv
    bias=False:
        bias 开启带来的收益与其面临的风险，不值当。
        DDPM 的每一步逆向操作，本质上都在“放大”信号。inv_sqrt_alpha 这是恒大于 1的。
        假设在 Epoch A，模型的最后一层 Bias 稍微偏大了一点点（比如 +0.005）
        模型倾向于预测“更正”的噪声，导致减去噪声后的 x_(t-1) 整体数值偏大。
        这个微小的偏差被 inv_sqrt_alpha 放大。
        经过 1000 步的累积，+0.005 的偏差可能变成了 +5.0 的偏差。
        最终图像数值全部 > 1，显示为全白。
        振荡：到了下一个 Epoch，Loss 发现预测结果太大了，于是疯狂惩罚参数，导致参数矫枉过正，Bias 变成了 -0.005。
        于是下一个 Epoch 采样出来就是全黑。

    关于在Conv前加一个激活函数？这个会改变数据的分布与均值，Act -> Norm -> Conv 也许可以，但为了简单，这里不使用激活函数。
    """

    def __init__(self,
                 in_channels:int,
                 time_emb_dim:int,
                 out_channels:int=3,
                 use_condconv:bool=True,
                 num_experts: int = 4):
        super().__init__()
        def make_conv(in_c, out_c, k, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=num_experts
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        # 使用 AdaCLN 做最后的归一化，确保去噪结束时的分布也受时间控制
        self.norm = AdaCLN(in_channels, time_emb_dim)
        self.conv = make_conv(in_c=in_channels, out_c=out_channels, k=3, s=1, p=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            x_in = self.norm(x_in, time_emb_in)
            return self.conv(x_in, time_emb_in)

        return main_path(x, time_emb)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # 只需要调节 scale，不需要 shift (RMSNorm 特性)
        # 初始化 weights 为 0，确保初始状态近似标准 RMSNorm
        # 分batch - 微调
        self.time_proj = TimeModulator(time_emb_dim, dim, 0)

        # 可学习的基础 scale，类似于 PyTorch RMSNorm 中的 weight
        # 全局 - 不分batch
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):

            # x: (B, N, C)
            # time_emb: (B, T_dim)

            # 1. 标准 RMSNorm 计算
            # x.pow(2).mean(-1) 计算均方值
            norm_x = x_in * torch.rsqrt(x_in.pow(2).mean(-1, keepdim=True) + self.eps)

            # 2. 应用基础 Scale
            norm_x = norm_x * self.scale

            # 3. 时间调制
            # (1 + scale_t) 保证初始状态下是一个 Identity 调制
            time_scale = self.time_proj(time_emb_in).unsqueeze(1)  # (B, 1, C)
            return norm_x * (1 + time_scale)

        return main_path(x, time_emb)


class TimeAwareSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, time_emb_dim):
        super().__init__()

        # 内容映射: 产生 Gate 的部分 和 Value 的部分
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_val = nn.Linear(dim, hidden_dim, bias=False)

        # 时间映射: 产生对 Gate 的控制信号
        self.w_time = TimeModulator(time_emb_dim, hidden_dim, 0)

        # 输出映射
        self.w_out = nn.Linear(hidden_dim, dim, bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            """
            x: (B, N, C)
            time_emb: (B, T_dim)
            """
            # 1. 投影内容
            c_gate = self.w_gate(x_in)  # (B, N, H)
            val = self.w_val(x_in)  # (B, N, H)

            # 2. 投影时间
            # 时间在这里充当一个全局的 Filter，对当前Batch的所有token作用一致
            t_gate = self.w_time(time_emb_in).unsqueeze(1)  # (B, 1, H)

            # 3. 时间感知门控 (Time-Aware Gating)
            gate = self.act(c_gate * (1 + t_gate))

            # 4. 应用门控并输出
            out = self.w_out(gate * val)
            return out

        return main_path(x, time_emb)


class TimeAwareTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 time_emb_dim,
                 resolution: tuple | int,
                 num_heads=8,
                 ffn_mult=4,
                 use_rope=True):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.use_rope = use_rope

        # Attention Norm
        self.norm1 = AdaRMSNorm(dim, time_emb_dim)

        # Self Attention
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        # FFN Norm
        self.norm2 = AdaRMSNorm(dim, time_emb_dim)

        # Time-Aware SwiGLU
        hidden_dim = int(dim * ffn_mult)
        self.ffn = TimeAwareSwiGLU(dim, hidden_dim, time_emb_dim)

        if use_rope:
            if isinstance(resolution, int):
                self.resolution = (resolution, resolution)
            else:
                self.resolution = resolution

            # 初始化 RoPE
            # 这里的 dim 必须对应 head_dim，而不是 channels
            self.rope = RoPE2D(
                dim=self.head_dim,
                height=self.resolution[0],
                width=self.resolution[1]
            )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        # x: (B, N, C), N = H*W
        B, N, C = x.shape
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            # Attention Block
            residual = x_in
            x_in = self.norm1(x_in, time_emb_in)

            # QKV
            qkv = self.attn_qkv(x_in).chunk(3, dim=-1)
            q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)

            # RoPE Injection
            if self.use_rope:
                # RoPE 期望输入: (B, H, SeqLen, HeadDim)
                q, k = self.rope(q, k)

            # Flash Attention
            x_in = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            x_in = x_in.transpose(1, 2).contiguous().view(B, N, C)
            x_in = self.attn_out(x_in)
            x_in = residual + x_in

            # Time-Aware FFN Block
            residual = x_in
            x_in = self.norm2(x_in, time_emb_in)
            x_in = self.ffn(x_in, time_emb_in)
            x_in = residual + x_in

            return x_in

        return main_path(x, time_emb)


class BottleneckTransformerStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 inner_dim: int,
                 num_layers: int,
                 time_emb_dim: int,
                 resolution: tuple | int,
                 num_heads: int,
                 ffn_mult: int=4,
                 use_condconv:bool=True):
        super().__init__()
        def make_conv(in_c, out_c, k=1, s=1, p=0, g=1):
            if use_condconv:
                return TimeAwareCondConv2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    time_emb_dim=time_emb_dim,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    groups=g,
                    bias=False,
                    num_experts=4
                )
            else:
                raise TypeError("This feature is not yet implemented.")

        self.proj_in = make_conv(in_channels, inner_dim, 1)

        self.layers = nn.ModuleList([
            TimeAwareTransformerBlock(
                dim=inner_dim,
                time_emb_dim=time_emb_dim,
                resolution=resolution,
                num_heads=num_heads,
                ffn_mult=ffn_mult
            )
            for _ in range(num_layers)
        ])

        # 融合层，加入原因不止是降维，还是Transformer与CNN的语言空间不同
        # 输入维度是 inner_dim * 2 (因为 concat 了 transformer 的输入和输出)
        # 或者 inner_dim + in_channels (如果你想 concat 原始输入)
        # 这里 concat (proj_in后的特征) 和 (transformer后的特征)，因为它们维度一致，语义层级接近
        self.proj_out = make_conv(inner_dim * 2, out_channels, 1)


        # 这里加一个可学习的缩放系数，初始化为很小的值，控制 Transformer 分支的权重
        # 在训练初期，它会发现 x_feat 这部分的通道更容易降低 Loss（恢复图像轮廓），所以它会赋予 x_feat 更高的权重。
        # 随着训练进行，Loss 到了瓶颈，单纯靠 x_feat 无法进一步降低 Loss 了（因为需要全局一致性），
        # 这时候 proj_out 会自动开始增加对 x_trans 部分通道的权重。
        self.trans_scale = nn.Parameter(torch.tensor(1e-4), requires_grad=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            # 1. 映射到隐空间
            # x_feat: (B, inner_dim, H, W)
            x_feat = self.proj_in(x_in, time_emb_in)

            # 2. 准备 Transformer 输入
            B, C, H, W = x_feat.shape
            # (B, N, C)
            x_trans = x_feat.flatten(2).transpose(1, 2)

            # 3. Transformer 处理
            for layer in self.layers:
                x_trans = layer(x_trans, time_emb_in)

            # 4. 还原回图像
            # (B, C, H, W)
            # 直接用 reshape (reshape会自动处理连续性)
            x_trans = x_trans.transpose(1, 2).reshape(B, C, H, W)

            # 5. 【核心策略】：Concat "ShortCut" 和 "Transformer Result"
            # 这样 Decoder 既能拿到 x_feat (保留了比较好的空间对应关系)
            # 又能拿到 x_trans (经过全局思考后的特征)

            # 显式地让 Transformer 分支从 0 开始学：
            x_trans = x_trans * self.trans_scale

            x_combined = torch.cat([x_feat, x_trans], dim=1)

            # 6. 融合并输出
            out = self.proj_out(x_combined, time_emb_in)

            return out

        if self.training:
            return checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            return main_path(x, time_emb)


class DiffusionUNet_64(nn.Module):
    def __init__(self, in_channels=3, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = TimeMLP(time_emb_dim=time_emb_dim, hidden_dim=512)

        # Stem
        self.stem = self.stem = Stem(in_channels, 64, time_emb_dim,
                 kernel_size=3, stride=1, padding=1, groups=1)

        # ================= ENCODER =================
        # Enc1: 64ch
        self.enc1 = DualPathStage(64, 64, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        # Down1: 112 -> 128
        self.down1 = DownsampleLayer(112, 128, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # Enc2: 128ch
        self.enc2 = DualPathStage(128, 128, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        # Down2: 176 -> 256
        self.down2 = DownsampleLayer(176, 256, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # Enc3: 256ch
        self.enc3 = DualPathStage(256, 256, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        self.att3 = TimeAwareSelfAttention(304, time_emb_dim, resolution=16, num_heads=4)  # 256 + 2*16 = 288
        # Down3: 304 -> 512
        self.down3 = DownsampleLayer(304, 512, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # ================= BOTTLENECK =================
        self.bot_stage = BottleneckTransformerStage(512, 512, 1024, num_layers=8,
                                                    time_emb_dim=time_emb_dim, resolution=8, num_heads=16)

        # ================= DECODER =================

        # --- Up 1 (8x8 -> 16x16) ---
        # Bot Out (512) -> Up (256)
        self.up1 = PixelShuffleUpsampleLayer(512, 256, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # Concat: Up1(256) + Enc3_Skip(304) = 560
        # Fusion: 560 -> 384
        self.fuse1 = FeatureFusionBlock(560, 384, time_emb_dim)

        # Stage: 处理 384 通道
        self.dec1 = DualPathStage(384, 384, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        self.dec1_att = TimeAwareSelfAttention(432, time_emb_dim, resolution=16, num_heads=6)  # 384 + 2*16 = 416

        # --- Up 2 (16x16 -> 32x32) ---
        # Dec1 Out (432) -> Up (192)
        self.up2 = PixelShuffleUpsampleLayer(432, 192, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # Concat: Up2(192) + Enc2_Skip(176) = 368
        # Fusion: 368 -> 192
        self.fuse2 = FeatureFusionBlock(368, 192, time_emb_dim)

        self.dec2 = DualPathStage(192, 192, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        # Dec2 Out: 240

        # --- Up 3 (32x32 -> 64x64) ---
        # Dec2 Out (240) -> Up (96)
        self.up3 = PixelShuffleUpsampleLayer(240, 96, dense_inc=0, time_emb_dim=time_emb_dim, groups=1)

        # Concat: Up3(96) + Enc1_Skip(112) = 208
        # Fusion: 208 -> 96
        self.fuse3 = FeatureFusionBlock(208, 96, time_emb_dim)

        self.dec3 = DualPathStage(96, 96, num_blocks=3, dense_inc=16, groups=4, time_emb_dim=time_emb_dim)
        # Dec3 Out: 144

        # ================= OUTPUT =================
        self.final = TimeAwareToRGB(144, time_emb_dim, 3)

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


# ================= 测试代码 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 1. 实例化模型
    model = DiffusionUNet_64().to(device)

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