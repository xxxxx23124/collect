import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint


class Factory:
    """
    统一管理 CondConv 和普通 Conv 的生成。
    统一管理 Transformer组件的生成。
    """
    def __init__(self, time_emb_dim, num_experts=4, attn_head_dim=64):
        self.time_emb_dim = time_emb_dim
        self.num_experts = num_experts
        # 旋转编码是公共的，实现在Factory中
        # 使用RoPE2D的默认配置，attn_head_dim为64就是默认配置
        self.attn_head_dim = attn_head_dim
        self.rope = RoPE2D(dim=self.attn_head_dim)
        self.get_conv_error = """
        作者是懒鬼，现在还没实现这个，如果要实现，作者会参考style_gan 2中权重调制，但需要注意，权重调制的成果会被CLN消除.
        权重调制，在作者眼中的为了放缩不同通道的重要性（不调制均值），与GRN很搭配（GRN看通道的能量，与均值有关），
        但CLN也是重新调制不同通道的重要性，但会重新归一化均值与方差，然后再次放缩，这会导致权重调制多余,
        因此这里还需要实现一个普通卷积。
        """

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
            num_experts=self.num_experts
        )
    def get_normconv(self,):
        raise NotImplementedError(self.get_conv_error)

    def get_convmod(self, ):
        raise NotImplementedError(self.get_conv_error)

    def get_act(self):
        # 卷积通路统一使用SiLU
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
    """
    2D 旋转位置编码 (Rotary Position Embedding)
    将 head_dim 的前半部分用于编码 Y(高度) 位置，后半部分用于编码 X(宽度) 位置。

    Args:
        dim: head dimension，必须是偶数
        height: 图像/特征图的高度（patch 数量）
        width: 图像/特征图的宽度（patch 数量）
        base: RoPE 的基础频率

    base 默认为500的解释：
                这里我们默认head dim为64为例子，也就是 16对分配给X，16对分配给Y。
                最低的频率为 500^(-15/16)，在8x8像素上，最远距离旋转 (7 * 500^(-15/16)) / pi * 180 大约 1.18 度
                最低的频率为 500^(-15/16)，在16x16像素上，最远距离旋转 (15 * 500^(-15/16)) / pi * 180 大约 2.53 度
                最低的频率为 500^(-15/16)，在32x32像素上，最远距离旋转 (31 * 500^(-15/16)) / pi * 180 大约 5.24 度
    """

    def __init__(
            self,
            dim:int=64,
            height:int=256,
            width:int=256,
            base:float=500.0
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

        # 5. 拉平为序列形式 -> (H*W, dim/2)
        freqs_y_full = emb_y_grid.reshape(-1, half_dim)
        freqs_x_full = emb_x_grid.reshape(-1, half_dim)

        # 6. 预计算并注册 cos/sin 缓存
        # shape: (1, 1, H*W, dim) 方便广播
        self.register_buffer("cos_y_cached", freqs_y_full.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_y_cached", freqs_y_full.sin().unsqueeze(0).unsqueeze(0))
        # shape: (1, 1, H*W, dim) 方便广播
        self.register_buffer("cos_x_cached", freqs_x_full.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_x_cached", freqs_x_full.sin().unsqueeze(0).unsqueeze(0))

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 query 和 key 应用 2D 旋转位置编码

        Args:
            q: Query tensor, shape (B, num_heads, SeqLen, head_dim)
            k: Key tensor, shape (B, num_heads, SeqLen, head_dim)

        Returns:
            tuple: 应用位置编码后的 (q, k)
        """
        seq_len = q.shape[2]

        def main_path(x_in: torch.Tensor):
            # 拆分 Y 部分和 X 部分
            # x: (B, H, N, D) -> y_part: (B, H, N, D/2), x_part: (B, H, N, D/2)
            y_part, x_part = x_in.chunk(2, dim=-1)

            # 获取 Y 部分对应长度的 cos/sin
            cos_y = self.cos_y_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)
            sin_y = self.sin_y_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)

            # 获取 X 部分对应长度的 cos/sin
            cos_x = self.cos_x_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)
            sin_x = self.sin_x_cached[:, :, :seq_len, :].to(dtype=x_in.dtype)

            # 分别应用 1D RoPE
            y_out = y_part * cos_y + self.rotate_half(y_part) * sin_y
            x_out = x_part * cos_x + self.rotate_half(x_part) * sin_x

            # 拼回去
            return torch.cat((y_out, x_out), dim=-1)


        q_embed = main_path(q)
        k_embed = main_path(k)

        return q_embed, k_embed

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将张量的后半部分旋转到前面，并取负"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) -> time steps
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TimeMLP(nn.Module):
    """
    将时间 t 映射为高质量的控制向量。
    """

    def __init__(self,
                 time_emb_dim=256,
                 hidden_dim=512,
                 num_layers=5,
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

    def forward(self, time: torch.Tensor) -> torch.Tensor:
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


class Modulator(nn.Module):
    """
    仅依赖时间 t，用于 AdaCLN, AdaRMS, SwiGLU。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(64, in_channels // 4)

        # 这个net的作用是独立调整，而非全面重构
        self.net = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, in_channels)
        )
        self.act = nn.SiLU()
        # bias=False 通常更好，防止引入特定的偏置倾向
        self.to_out = nn.Linear(in_channels, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        # 初始化为0，保证初始状态下不影响主干流
        nn.init.zeros_(self.to_out.weight)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        time_emb = time_emb + self.net(time_emb)
        time_emb = self.act(time_emb)
        return self.to_out(time_emb)


class ContentAwareModulator(nn.Module):
    """
    同时依赖时间 t 和 内容 x (Mean + Std)。
    用于 CondConvRouter, AttentiveAdaGRN。
    """

    def __init__(self,
                 in_channels,
                 time_emb_dim,
                 out_channels,
                 hidden_dim=None,
                 eps=1e-6):
        super().__init__()

        # 如果不指定 hidden_dim，就计算一个合适的大小
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(64, in_channels // 4)
        self.eps = eps
        # 1. Q 映射 (Time)
        self.to_q = nn.Linear(time_emb_dim, self.hidden_dim, bias=False)

        # 2. K 映射 (Content: Mean + Std) -> 输入是 2 * in_channels
        self.to_k = nn.Linear(in_channels * 2, self.hidden_dim, bias=False)

        # 3. V 映射 (Content: Mean + Std)
        self.to_v = nn.Linear(in_channels * 2, self.hidden_dim, bias=False)

        # 4. Mixer
        self.mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 5. Output Projector (Zero Init)
        # bias=False 通常更好，防止引入特定的偏置倾向
        self.to_out = nn.Linear(self.hidden_dim, out_channels, bias=False)

        self._init_weights()

    def _init_weights(self):
        # 保持零初始化策略
        # 对于 Router，这意味初始概率均匀 (logits全是0 -> softmax后均匀)
        # 对于 GRN，这意味着初始为 Identity
        nn.init.zeros_(self.to_out.weight)

    def _compute_stats(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Mean
        mu = x.mean(dim=(2, 3))
        # Std (加上 eps 防止 NaN)
        """
        unbiased=False（有偏）：
            分母使用 N （样本总数）
            这是描述当前这组数据本身的标准差。
        unbiased=True（无偏）：
            分母使用 N-1
            这是统计学中当我们只拥有一小部分样本，
            想要去估计整个总体（Population）的标准差时使用的公式。
            减去 1 是为了校正偏差（贝塞尔校正）。
        """
        std = x.std(dim=(2, 3), unbiased=False) + self.eps
        return torch.cat([mu, std], dim=1)  # (B, 2C)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) - 需要完整的特征图来计算 Std
        time_emb: (B, T_dim)
        """
        # 1. 提取统计量 (Perception)
        stats = self._compute_stats(x)  # (B, 2C)

        # 2. Attention 机制
        q = self.to_q(time_emb)  # (B, H)
        k = self.to_k(stats)  # (B, H)
        v = self.to_v(stats)  # (B, H)

        attn = torch.sigmoid(q * k)  # 门控
        x_hidden = v * attn  # 筛选特征

        # 3. 混合推理
        x_mixed = x_hidden + self.mixer(x_hidden)

        # 4. 输出
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
        """
        x: (B, C, H, W)
        time_emb: (B, T_dim)
        """
        return self.modulator(x, time_emb)  # (B, num_experts)


class TimeAwareCondConv2d(nn.Module):
    """
    Time-Aware CondConv: Dynamically aggregates expert kernels using both input features and time embeddings.
    Allows the convolution to adapt its behavior based on the current diffusion timestep (t).
    Maintains standard Conv2d interface while supporting groups and efficient batch processing.


    关于添加噪声：
        如 F.softmax(routing_logits + N(0,1), dim=1)
        不添加任何人为噪音可能是适合DDPM的。

    关于Softmax vs Sigmoid：
        softmax(routing_logits) or sigmoid(routing_logits)
       用softmax可能更适合DDPM。原本的sigmoid激活在图像分类/检测任务中可能影响不大
       （例如当出现某种模式时，就是那个类别，检测任务中的定位的精确度也不像DDPM这么严格），
       但在DDPM这一类要求精确的回归任务中，如果方差不可控，模型可能会消耗很多精力去适应sigmoid组合专家带来的统计分布变化。
       这会导致严重的“内耗”。
       补充：
            经过不严谨的实验（没有专门控制变量），貌似使用Sigmoid激活，会让模型在前期采样的图容易变得全是黑的，或者全是白的（大概将近1/2都是全白全黑）。
            使用Softmax的，全白全黑的现象少了很多（20个Epoch的采样，每个Epoch有105步左右，只出现了一次全白）

    关于正则化：
       目前决定不启用正则化强制让专家负载均衡。
       原因：卷积核不像大语言模型中的专家（MLP），卷积核的参数容量很小，主要是做模式匹配。
       在DDPM中，初期步（构图）和后期步（去噪）的任务差距很大，不同的卷积核天然倾向于负责不同阶段的任务。

       目前添加得有使得专家正交的正则化，即专家间的知识不同。

    实际测试 4专家 softmax(routing_logits)：
        无任何正则化，包括专家正交的正则化：
            在简单数据集，如64像素猫头，会出现专家死亡
            在复杂数据集，如64像素ImageNet，没有出现专家死亡，
            但在encoder的enc2中，可能因为难度不够大，导致可能只利用2个专家，其余专家保持1e-4级别的利用率。
        使用了专家正交正则化：
            ortho_weight=1e-5的情况下，在64像素猫头上，确实解决了专家死亡的问题，最低利用率在1%左右，
            绝大部分都启用了4个专家，少数启用3个专家，少数启用2个专家。
            数据说话就是，熵最低的也是在0.8左右波动，113个中只有1个。
            熵在1左右波动的有3个，
            观察到一个现象，部分专家间竞争震荡比较严重（ortho_weight=1e-5可能太大了，且学习率2e-4也太大了），利用率少的专家，几乎无震荡，且始终保持了1%的利用率。
            经过测试，ortho_weight=1e-6，lr=1e-4这一组超参数不错，有效遏制专家死亡的前提下稳定了训练。
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

        # Expert weights: [num_experts, out_channels, in_channels//groups, k, k]
        self.weight = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels // groups, self.kernel_size, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.randn(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        # Routing network: MLP taking both (AvgPooled Input + Time Embedding)
        self.router = CondConvRouter(in_channels, time_emb_dim, num_experts)

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

    def get_ortho_loss(self):
        """
        计算正交正则化损失。
        目标：让不同专家的卷积核方向尽可能垂直（不相似）。
        """
        # weight shape: [num_experts, out, in/g, k, k]
        n_exp = self.num_experts

        # 展平每个专家的参数向量
        # [num_experts, N_params]
        w_flat = self.weight.view(n_exp, -1)

        # 归一化 (只看方向，不看模长)
        w_norm = F.normalize(w_flat, p=2, dim=1)

        # 计算 Gram 矩阵 (余弦相似度矩阵)
        # [num_experts, num_experts]
        gram = torch.mm(w_norm, w_norm.t())

        # 目标是单位矩阵 Identity (对角线为1，其余为0)
        # 减去对角线部分，只惩罚非对角线元素 (即不同专家之间的相似度)
        identity = torch.eye(n_exp, device=w_flat.device)

        # Frobenius 范数
        ortho_loss = torch.norm(gram - identity, p='fro')

        return ortho_loss

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature map [B, C, H, W].
            time_emb (Tensor): Time embedding vector [B, D] indicating noise level.
        Returns:
            Tensor: Output feature map computed with time-adaptive kernels.
        """

        B, C, H, W = x.shape
        # Calculate routing weights based on Content + Time
        # 使用 Softmax 保证权重之和为 1，维持特征图的方差稳定性
        routing_logits = self.router(x, time_emb)
        routing_weights = F.softmax(routing_logits, dim=1) # [B, num_experts]

        # Compute effective weights by aggregating experts: Sum(weight_i * routing_i)
        weight_eff = (routing_weights.view(B, self.num_experts, 1, 1, 1, 1) *
                      self.weight.unsqueeze(0)).sum(1)  # [B, out, in/g, k, k]

        # Compute effective bias if present
        if self.bias is not None:
            bias_eff = (routing_weights.view(B, self.num_experts, 1) *
                        self.bias.unsqueeze(0)).sum(1)  # [B, out]
            bias_eff = bias_eff.view(-1)  # Flatten to [B * out] for conv2d
        else:
            bias_eff = None

        # Flatten batch and channels for grouped convolution trick
        x_flat = x.view(1, B * C, H, W)  # [1, B*in, H, W]
        # Flatten weights: [B * out, in/g, k, k]
        weight_flat = weight_eff.view(B * self.out_channels,
                                      self.in_channels // self.groups,
                                      self.kernel_size,
                                      self.kernel_size)
        # Calculate groups for the single large convolution
        total_groups = B * self.groups

        # Apply convolution with unique kernel per sample
        out_flat = F.conv2d(
            x_flat,
            weight_flat,
            bias_eff,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=total_groups
        )  # [1, B*out, H', W']

        # Reshape back to standard batch format [B, out, H', W']
        return out_flat.view(B, self.out_channels, out_flat.shape[2], out_flat.shape[3])


class AdaCLN(nn.Module):
    """
    Adaptive Channel Layer Norm (AdaCLN)
    Replaces static weight/bias with time-dependent modulation.
    """

    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.num_channels = channels
        self.eps = eps

        # 将时间嵌入映射为 Scale (gamma) 和 Shift (beta)
        self.emb_layers = Modulator(time_emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
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


class AdaGRN(nn.Module):
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 这一部分用来生成 gamma 和 beta
        # 输出维度 = 2 * channels (Gamma, Beta)
        # hidden_dim 可以设为 channels // 4 或者直接 64，视通道数大小而定
        self.modulator = ContentAwareModulator(
            in_channels=channels,
            time_emb_dim=time_emb_dim,
            out_channels=2 * channels,
            hidden_dim=max(64, channels // 4)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1. 标准 GRN 计算 (竞争归一化)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        # 2. 生成动态参数 (Mean+Std + Time -> Gamma/Beta)
        params = self.modulator(x, time_emb)
        gamma, beta = params.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # 3. 应用 (Residual Style)
        # 初始时 gamma, beta 为 0，退化为 x，保证训练稳定
        # GRN 项: (x * nx) 代表竞争后的特征
        return x * (1 + gamma * nx) + beta


class AdaRMSNorm(nn.Module):
    """
    统一的 AdaRMSNorm，只支持 (B, N, C)
    """
    def __init__(self, channels, time_emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = channels

        # 只需要调节 scale，不需要 shift (RMSNorm 特性)
        # TimeModulator初始输出恒定为0
        # 分batch - 微调
        self.time_proj = Modulator(time_emb_dim, channels)

        # 可学习的基础 scale，类似于 PyTorch RMSNorm 中的 weight
        # 全局 - 不分batch
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        # time_emb: (B, T_dim)
        """

        # 标准 RMSNorm 计算
        # x.pow(2).mean(-1) 计算均方值
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 应用基础 Scale
        norm_x = norm_x * self.scale

        # 时间调制
        # (1 + scale_t) 保证初始状态下是一个 Identity 调制
        time_scale = self.time_proj(time_emb).unsqueeze(1)  # (B, 1, C)
        out = norm_x * (1 + time_scale)

        return out


class SelfAttention(nn.Module):
    """
    核心注意力机制，不关心输入是图片还是序列，只处理 (B, N, C) 格式。
    默认包含 RoPE 和 Time-Aware QKV 投影。
    现阶段不支持窗口注意力需要的位置平移。
    注意：这个模块依赖于RoPE2D中的dim成员变量
    """

    def __init__(self,
                 channels:int,
                 num_heads:int,
                 rope: RoPE2D):
        super().__init__()
        self.num_heads = num_heads
        # 注意，这里的head_dim来自rope中的dim，
        # 这样写是为了防止出错，只需要配置多少个头就好
        head_dim = rope.dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        inner_dim = num_heads * head_dim

        self.to_qkv = nn.Linear(channels, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, channels, bias=False)

        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        B, N, C = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, N, heads, dim) -> (B, heads, N, dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 这里默认应用旋转编码，这里不用担心序列长度的问题，因为在rope会自动检测
        # 目前默认为 256*256 像素，不超过 256*256 像素都行
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

        # 内容映射: 产生 Gate 的部分 和 Value 的部分
        self.w_gate = nn.Linear(channels, inner_channels, bias=False)
        self.w_val = nn.Linear(channels, inner_channels, bias=False)

        # 时间映射: 产生对 Gate 的控制信号
        self.w_time = Modulator(time_emb_dim, inner_channels)

        # 输出映射
        self.w_out = nn.Linear(inner_channels, channels, bias=False)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor)-> torch.Tensor:
        """
        x: (B, N, C)
        time_emb: (B, T_dim)
        """
        # 1. 投影内容
        c_gate = self.w_gate(x)  # (B, N, H)
        val = self.w_val(x)  # (B, N, H)

        # 2. 投影时间
        # 时间在这里充当一个全局的 Filter，对当前Batch的所有token作用一致
        t_gate = self.w_time(time_emb).unsqueeze(1)  # (B, 1, H)

        # 3. 时间感知门控 (Time-Aware Gating)
        # 往往一些数值微小的差异，或者比例的差异，就会让状态完全不同
        # 只需要在关键的临界点（Gate）上推一把，整个网络的推理路径就会发生分叉
        gate = self.act(c_gate * (1 + t_gate))

        # 4. 应用门控并输出
        out = self.w_out(gate * val)
        return out


class DualPathBlock(nn.Module):
    """
    ConvNeXt-like Dual Path Block: Combines residual and dense connections.

    inspired by ConvNeXt V1/V2:
        Larger 7x7 kernel for spatial mixing.
        Inverted bottleneck MLP in pointwise part (1x1 up -> SiLU -> GRN -> 1x1 down).
        Minimal Channel-wise LayerNorm.
        GELU activation.
        GRN from ConvNeXt V2 to prevent feature collapse.

    Reference:
        https://arxiv.org/pdf/1707.01629
        https://arxiv.org/pdf/2201.03545
        https://arxiv.org/pdf/2301.00808
    """
    def __init__(self,
                 in_channels: int,
                 res_channels: int,
                 dense_inc: int,
                 groups: int,
                 factory: Factory,
                 expansion_ratio: int=4, ):
        super().__init__()
        assert res_channels % groups == 0, f"res_channels:{res_channels} must be divisible by groups:{groups}."

        # 这个变量在forward中有用
        self.res_channels = res_channels

        # conv1 的作用是对主干的洪流进行降维，1x1卷积，
        # 这里关于bias的设置，我是认为：可以开，也可以关闭，
        # 因为后面没有直接连接的cln，bias不会白费，也没有grn。
        # 但这这里，我选择关闭bias，经验主义，非要接受的话，就是：
        # 模型通道的底色（bias带来的均值）不是由bias调制的，是由时间向量调制的。
        self.conv1 = factory.get_condconv(in_channels=in_channels,
                                          out_channels=res_channels,
                                          kernel_size=1,
                                          bias=False)

        # 这是7x7卷积，因为后面紧接着cln，bias带来的均值影响被彻底抹除，开启bias就是浪费
        # 这里补充一个思考 7x7卷积就像是一种窗口注意力，但不是两个像素间的注意力，而是像素（token）与卷积核的注意力（相似度，匹配度）
        self.conv2 = factory.get_condconv(in_channels=res_channels,
                                          out_channels=res_channels,
                                          kernel_size=7,
                                          padding=3,
                                          groups=groups,
                                          bias=False)

        # CLN
        self.cln = factory.get_cln(res_channels)

        mid_channels = res_channels * expansion_ratio
        # 必须关闭bias（这里bias是默认关闭的），因为bias会影响接下来的GRN工作
        # 对于conv3，这里是记忆（KV对）的存储中心的前半段，将7x7的输入，按照自己的知识转换为有用的输出
        # 在相对简单的任务下，在encoder中，这个模块的压力小，condconv显得杀鸡用牛刀
        # 这里为了代码简单，统一使用condconv
        self.conv3_up = factory.get_condconv(in_channels=res_channels,
                                             out_channels=mid_channels,
                                             kernel_size=1,
                                             bias=False)

        self.act = factory.get_act()

        # GRN
        self.grn = factory.get_grn(mid_channels)

        # 这里是记忆存储中心的后半段
        # 因为这两个输出之后，可能就是残差或者拼接，加的bias可能在后面模块的cln面前没意义，意义不大
        # 这个地方需要主意，我没有一次输出res + dense 两部分
        # 因为res和dense可能负担的任务不同
        # 因此他们的专家调用情况可能不同，因此这两个模块用有两个路由，而不是共用一个路由
        # 这两个1x1卷积，在encoder中，同样可以用非condconv替代
        self.conv3_res = factory.get_condconv(in_channels=mid_channels,
                                              out_channels=res_channels,
                                              kernel_size=1,
                                              bias=False)
        self.conv3_dense = factory.get_condconv(in_channels=mid_channels,
                                                out_channels=dense_inc,
                                                kernel_size=1,
                                                bias=False)

        # LayerScale 参数
        # gamma_res 对应 new_res 的通道数 (res_channels)
        # 初始化为一个非常小的数 (epsilon)，例如 1e-6
        # 这样初始时，new_res * gamma_res ≈ 0，相当于恒等映射
        self.gamma_res = nn.Parameter(1e-6 * torch.ones(res_channels), requires_grad=True)

        # 对于 Dense 部分，小一些是为了稳定训练，小一些，dense连接对后层的方差影响比较小
        # 需要比gamma_res大，以防止Dense路径失活
        self.gamma_dense = nn.Parameter(1e-2 * torch.ones(dense_inc), requires_grad=True)

    def forward(self, x, time_emb) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
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
    """
    A stage of DualPathBlocks.

    This module sequentially applies a specified number of DualPathBlocks.
    Its primary responsibility is to deepen the network representation at a
    constant spatial resolution.
    """

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
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor)-> torch.Tensor:

            for block in self.blocks:
                x_in = block(x_in, time_emb_in)
            return x_in

        if self.training:
            return checkpoint(main_path, x, time_emb, use_reentrant=False)
        else:
            return main_path(x, time_emb)


class DownsampleLayer(nn.Module):
    """
    DownsampleLayer for DPN: Downsamples feature maps and adjusts channel dimensions.
    ConvNeXt-like: LN + 2x2 strided conv.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 factory: Factory,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_inc = dense_inc

        self.cln = factory.get_cln(in_channels)
        # 这里有一个注意事项是为了追求性能，不能用分组卷积
        # 这里选择关闭bias，因为后面是残差连接，后面可能连接cln，因此就不开bias了
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
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        x = self.cln(x, time_emb)
        return self.conv(x, time_emb)


class PixelShuffleUpsampleLayer(nn.Module):
    """
    追求质量的 PixelShuffle 上采样层。
    结构: LN -> AdaCLN -> Conv(in, out*4) -> PixelShuffle -> (concat -> AdaCLN -> SiLU)

    以下是对这个模块中的一些超参数设置的解释：
        TimeAwareCondConv2d 的 bias=False：
            卷积层的 Bias 是加在通道上的。
            假设输出通道是 4 个，Bias向量为[b_0, b_1, b_2, b_3]
            如果 b_0 很亮，而 b_1 很暗。
            那么无论输入图像是什么，输出图像的所有2x2像素块里，左上角永远比右上角亮。
            这相当于在整张图上叠加了一层永久去不掉的网格纹理。

    为什么这里面没有激活函数呢？因为PixelShuffleUpsampleLayer的下一个模块就是FeatureFusionBlock，
    在FeatureFusionBlock是有激活函数的。
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dense_inc: int,
                 factory: Factory, ):
        super().__init__()

        # 目标输出通道数
        target_out = out_channels + dense_inc
        # PixelShuffle 需要 r^2 倍的通道数，这里 scale=2，所以是 4 倍
        expansion = 4

        # 内部卷积输出通道数
        inner_out = target_out * expansion

        self.cln = factory.get_cln(in_channels)

        # 这里我们使用 3x3 卷积来生成扩展通道，增加感受野
        # 这里关闭bias的原因就是希望放大时不用带偏见
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

        # 如果是 CondConv，就进行 ICNR 初始化
        # 这是一个非常关键的步骤，防止 PixelShuffle 初始化时的棋盘格效应
        if isinstance(self.conv, TimeAwareCondConv2d):
            # 对于 CondConv，我们对每个 expert 进行 ICNR 初始化
            for i in range(self.conv.num_experts):
                self._icnr_init(self.conv.weight[i], scale=2)

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

    def forward(self, x, time_emb)-> torch.Tensor:
        x = self.cln(x, time_emb)
        # 先卷积增加通道 (低分辨率)
        x = self.conv(x, time_emb)
        # 再 Shuffle 变大尺寸
        x = self.pixel_shuffle(x)
        return x


class FeatureFusionBlock(nn.Module):
    """
    专门用于 Decoder 阶段 skip connection 拼接后的特征融合与降维。
    结构: AdaCLN -> SiLU -> 1x1 TimeAwareCondConv

    为什么要加激活函数呢？
    这是我的直觉，我觉得通道数是足够传递信息的（应该还冗余），不用担心激活损坏信息，
    同时受到MaxPooling的启发，也许加一个激活，保留下强的信号，过滤弱的信号，可以有利于模型学习更鲁棒的特征

    这个模块是Decoder中的DualPathStage的开始，也就意味着这个模块是残差直连到底的，就不开bias了
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 factory: Factory,):
        super().__init__()
        self.cln = factory.get_cln(in_channels)
        self.act = factory.get_act()
        # 1x1 卷积用于降维，不改变空间尺寸
        # 不要使用分组卷积
        self.conv = factory.get_condconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor)-> torch.Tensor:
        x = self.cln(x, time_emb)
        x = self.act(x)
        return self.conv(x, time_emb)


class SpatialSelfAttention(nn.Module):
    """
    作用于CNN的注意力
    """

    def __init__(self,
                 channels: int,
                 num_heads: int,
                 factory: Factory,):
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
        """
        x: (B, C, H, W)
        time_emb: (B, t_dim)
        """
        B, C, H, W = x.shape

        residual = x
        # RMSNorm
        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        x_norm = self.rms(x_flat, time_emb)
        # Attention
        out = self.attn(x_norm)
        # Reshape back & Residual
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return residual + out


class TransformerBlock(nn.Module):
    def __init__(self,
                 channels:int,
                 factory: Factory,
                 num_heads:int=8,
                 ffn_mult:float=4.0, ):
        super().__init__()
        self.num_heads = num_heads

        # Attention Norm
        self.rms1 = factory.get_rms(channels=channels)

        # Self Attention
        self.attn = factory.get_selfattn(channels=channels, num_heads=num_heads)

        # FFN Norm
        self.rms2 = factory.get_rms(channels=channels)

        # Time-Aware SwiGLU
        inner_channels = int(channels * ffn_mult)
        self.ffn = factory.get_swiglu(channels=channels, inner_channels=inner_channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor)-> torch.Tensor:
        # Attention Part
        x = x + self.attn(self.rms1(x, time_emb))
        # FFN Part
        x = x + self.ffn(self.rms2(x, time_emb), time_emb)
        return x


class BottleneckTransformerStage(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 inner_channels:int,
                 num_layers:int,
                 num_heads:int,
                 factory: Factory,
                 ffn_mult:float=4.0,):
        super().__init__()

        # 这个不要开启bias，即影响下面的RMSNorm，又影响之后的GRN
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

        # 融合层，加入原因不止是降维，还是Transformer与CNN的语言空间不同
        self.fusion_act = factory.get_act()
        # GRN鼓励通道间的竞争，即让transformer和proj_in竞争
        """
        GRN补充[来自ConvNeXt V2]：
            GRN（Global Response Normalization）被提出的核心目的，
            是为了解决通道间的特征坍塌（Feature Collapse）问题，
            从而增加通道间信息的多样性，避免冗余（即大量通道失活或表现一致）
        """
        self.fusion_grn = factory.get_grn(inner_channels * 2)
        # 输入维度是 inner_dim * 2 (因为 concat 了 transformer 的输入和输出)
        # 或者 inner_dim + in_channels (如果你想 concat 原始输入)
        # 这里 concat (proj_in后的特征) 和 (transformer后的特征)，因为它们维度一致，语义层级接近
        # 这里的下一个模块应该就是 PixelShuffleUpsampleLayer，立马就会遇到CLN，因此开启bias也没有意义。
        self.proj_out = factory.get_condconv(
            in_channels=inner_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )

        # 这里加一个可学习的缩放系数，初始化为很小的值，控制 Transformer 分支的权重
        # 因为GRN的存在，transformer最开始是被压制的，因为trans_scale让transformer闭嘴了
        # 当transformer开始说话时，就是模型需要的信息，而不是那些显而易见的信息
        # 这个时候，原始的通道的连接提供了修复/还原被transformer损失的信息
        # 这时，这个transformer应该就可以比较好地融入这个CNN架构（因为transformer的工作方式天生和CNN不同，同一个dim内的语义是不同的）
        self.trans_scale = nn.Parameter(torch.tensor(1e-2), requires_grad=True)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor)-> torch.Tensor:
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor)-> torch.Tensor:
            # 1. 映射到隐空间
            # x_feat: (B, inner_dim, H, W)
            x_feat = self.proj_in(x_in, time_emb_in)

            # 2. 准备 Transformer 输入
            B, C, H, W = x_feat.shape
            # (B, N, C)
            x_trans = x_feat.flatten(2).transpose(1, 2).contiguous()

            # 3. Transformer 处理
            for layer in self.layers:
                x_trans = layer(x_trans, time_emb_in)

            # 4. 还原回图像
            # (B, C, H, W)
            # 直接用 reshape (reshape会自动处理连续性)
            x_trans = x_trans.transpose(1, 2).reshape(B, C, H, W)

            # 5. Concat "ShortCut" 和 "Transformer Result"
            # 这样 Decoder 既能拿到 x_feat (保留了比较好的空间对应关系)
            # 又能拿到 x_trans (经过全局思考后的特征)

            # 显式地让 Transformer 分支从 0 开始学：
            x_trans = x_trans * self.trans_scale
            x_combined = torch.cat([x_feat, x_trans], dim=1)
            # 激活筛选过滤信息
            x_combined = self.fusion_act(x_combined)
            # GRN鼓励通道间竞争
            x_combined = self.fusion_grn(x_combined, time_emb_in)

            # 6. 融合并输出
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
                 factory: Factory,):

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
        """
        Args:
            x(torch.Tensor): Input Tensor
            time_emb(torch.Tensor): Timestep Tensor
        Return:
            Tensor
        """
        out = self.conv(x, time_emb)
        out = self.cln(out, time_emb)
        return out


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
                 out_channels:int,
                 factory: Factory,):
        super().__init__()
        assert out_channels == 3, "out_channels should be 3."
        # 使用 AdaCLN 做最后的归一化，确保去噪结束时的分布也受时间控制
        self.cln = factory.get_cln(channels=in_channels)
        self.conv =         factory.get_condconv(
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
        factory = Factory(time_emb_dim=time_emb_dim, num_experts=4, attn_head_dim=64)

        # Stem
        self.stem = self.stem = Stem(in_channels=in_channels, out_channels=64, factory=factory)

        # ================= ENCODER =================
        # Enc1: 64ch
        self.enc1 = DualPathStage(in_channels=64, res_channels=64, num_blocks=3, dense_inc=16, groups=4, factory=factory)
        # Down1: 112 -> 128
        self.down1 = DownsampleLayer(in_channels=112, out_channels=128, dense_inc=0, factory=factory)

        # Enc2: 128ch
        self.enc2 = DualPathStage(in_channels=128, res_channels=128, num_blocks=3, dense_inc=16, groups=4, factory=factory)
        # Down2: 176 -> 256
        self.down2 = DownsampleLayer(in_channels=176, out_channels=256, dense_inc=0, factory=factory)

        # Enc3: 256ch
        self.enc3 = DualPathStage(in_channels=256, res_channels=256, num_blocks=4, dense_inc=16, groups=4, factory=factory)
        self.att3 = SpatialSelfAttention(channels=320, num_heads=5, factory=factory)
        # Down3: 320 -> 512
        self.down3 = DownsampleLayer(in_channels=320, out_channels=512, dense_inc=0, factory=factory)

        # ================= BOTTLENECK =================
        self.bot_stage = BottleneckTransformerStage(in_channels=512, out_channels=512, inner_channels=1024, num_layers=10,
                                                    num_heads=16, factory=factory)

        # ================= DECODER =================

        # --- Up 1 (8x8 -> 16x16) ---
        # Bot Out (512) -> Up (256)
        self.up1 = PixelShuffleUpsampleLayer(in_channels=512, out_channels=256, dense_inc=0, factory=factory)

        # Concat: Up1(256) + Enc3_Skip(320) = 576
        # Fusion: 560 -> 384
        self.fuse1 = FeatureFusionBlock(in_channels=576, out_channels=384, factory=factory)

        # Stage: 处理 384 通道
        self.dec1 = DualPathStage(in_channels=384, res_channels=384, num_blocks=4, dense_inc=16, groups=4, factory=factory)
        self.dec1_att = SpatialSelfAttention(channels=448, num_heads=7, factory=factory)

        # --- Up 2 (16x16 -> 32x32) ---
        # Dec1 Out (448) -> Up (192)
        self.up2 = PixelShuffleUpsampleLayer(in_channels=448, out_channels=192, dense_inc=0, factory=factory)

        # Concat: Up2(192) + Enc2_Skip(176) = 368
        # Fusion: 368 -> 192
        self.fuse2 = FeatureFusionBlock(in_channels=368, out_channels=192, factory=factory)

        self.dec2 = DualPathStage(in_channels=192, res_channels=192, num_blocks=3, dense_inc=16, groups=4, factory=factory)
        # Dec2 Out: 240

        # --- Up 3 (32x32 -> 64x64) ---
        # Dec2 Out (240) -> Up (96)
        self.up3 = PixelShuffleUpsampleLayer(in_channels=240, out_channels=96, dense_inc=0, factory=factory)

        # Concat: Up3(96) + Enc1_Skip(112) = 208
        # Fusion: 208 -> 96
        self.fuse3 = FeatureFusionBlock(in_channels=208, out_channels=96, factory=factory)

        self.dec3 = DualPathStage(in_channels=96, res_channels=96, num_blocks=3, dense_inc=16, groups=4, factory=factory)
        # Dec3 Out: 144

        # ================= OUTPUT =================
        self.final = TimeAwareToRGB(in_channels=144, out_channels=out_channels, factory=factory)

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

        # 遍历模型中所有的 TimeAwareCondConv2d
        for module in self.modules():
            if isinstance(module, TimeAwareCondConv2d):
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