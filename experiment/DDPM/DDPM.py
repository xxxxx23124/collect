import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self,
                 model,
                 timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 use_fixed_small=True,):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.use_fixed_small = use_fixed_small

        # ============================================================
        # 1. 预计算扩散过程所需的常数
        # ============================================================
        # 先在 CPU 上计算好所有数值，最后统一注册
        # 当对 ddpm 实例调用 .to(device) 时，它们会自动移动

        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        # 注册为 buffer (会自动加入 state_dict，并随模型移动设备)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

        # 预计算常用的平方根
        self.register_buffer("sqrt_alpha_hat", torch.sqrt(alpha_hat))
        self.register_buffer("sqrt_one_minus_alpha_hat", torch.sqrt(1.0 - alpha_hat))

        # 反向过程所需的常数
        # mu计算中的系数
        inv_sqrt_alpha = 1.0 / torch.sqrt(alpha)
        denoise_coeff = beta / torch.sqrt(1.0 - alpha_hat)

        self.register_buffer("inv_sqrt_alpha", inv_sqrt_alpha)
        self.register_buffer("denoise_coeff", denoise_coeff)

        # 计算 alpha_hat_prev (即 alpha_hat[t-1])
        # 对于 t=0, alpha_hat_prev 应为 1.0
        # 将 alpha_hat 向右移一位，并在第0位填充 1.0
        alpha_hat_prev = F.pad(alpha_hat[:-1], (1, 0), value=1.0)
        self.register_buffer("alpha_hat_prev", alpha_hat_prev)

        posterior_variance = beta * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)
        # 为了数值稳定性，clamp一下，为了防止数值误差导致出现极小的负数，clamp 最小值为 0
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=0.0))

        # Sigma (方差)
        # Fixed Large
        # 当 t=0 时，sigma_fl 不为 0。
        self.register_buffer("sigma_fl", torch.sqrt(beta))
        # Fixed Small
        # 当 t=0 时，sigma_fs 为 0。
        self.register_buffer("sigma_fs", torch.sqrt(posterior_variance.clamp(min=0.0)))

    @property
    def device(self):
        return self.beta.device

    def extract(self, a, t, x_shape):
        """
        辅助函数：从长为 T 的一维张量 a 中提取对应时间步 t 的值，
        并reshape成 (Batch, 1, 1, 1) 以便与图像做广播运算。
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_0, t):
        """
        前向加噪过程：给定 x_0 和 t，生成 x_t 和对应的噪声 epsilon
        公式: x_t = sqrt(alpha_hat) * x_0 + sqrt(1-alpha_hat) * epsilon
        """
        # 生成随机噪声
        noise = torch.randn_like(x_0)

        # 获取当前 t 对应的系数
        sqrt_alpha_hat_t = self.extract(self.sqrt_alpha_hat, t, x_0.shape)
        sqrt_one_minus_alpha_hat_t = self.extract(self.sqrt_one_minus_alpha_hat, t, x_0.shape)

        # 加噪
        x_t = sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

        return x_t, noise

    def compute_loss(self, x_0):
        """
        计算 DDPM 的损失
        x_0: 原始图像 (Batch, C, H, W)，假设范围已经归一化到 [-1, 1]
        """
        batch_size = x_0.shape[0]

        # 1. 随机采样时间步 t ~ Uniform(0, T-1)
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        # 2. 生成噪声图 x_t 和 真实噪声 noise
        x_t, noise = self.add_noise(x_0, t)

        # 3. 神经网络预测噪声
        t_normalized = t.float() / self.timesteps
        # 用了 GaussianFourierProjection，所以 必须归一化
        predicted_noise = self.model(x_t, t_normalized)

        # 4. 计算 MSE Loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, img_size, channels=3):
        """
        采样过程 (Reverse Process): 从纯噪声生成图像
        """
        self.model.eval()

        # 1. 从标准正态分布采样 x_T
        x = torch.randn(num_samples, channels, img_size, img_size).to(self.device)

        # 2. 从 T-1 倒推到 0
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = (torch.ones(num_samples) * i).long().to(self.device)

            # 预测噪声
            t_normalized = t.float() / self.timesteps
            # 用了 GaussianFourierProjection，所以 必须归一化
            predicted_noise = self.model(x, t_normalized)

            # 计算均值 mu
            # 公式: x_{t-1} = 1/sqrt(alpha) * (x_t - (beta / sqrt(1-alpha_hat)) * eps)
            # 直接提取预计算好的参数
            inv_sqrt_alpha_t = self.extract(self.inv_sqrt_alpha, t, x.shape)
            denoise_coeff_t = self.extract(self.denoise_coeff, t, x.shape)

            # 计算 mu
            mu = inv_sqrt_alpha_t * (x - denoise_coeff_t * predicted_noise)

            # 如果不是最后一步 (t > 0)，则添加噪声
            if i > 0:
                noise = torch.randn_like(x)
                if self.use_fixed_small:
                    sigma_t = self.extract(self.sigma_fs, t, x.shape)
                else:
                    sigma_t = self.extract(self.sigma_fl, t, x.shape)
                x = mu + sigma_t * noise
            else:
                # 最后一步 (t=0 -> x_0) 不加噪声
                # 理论上 posterior_std[0] 也是 0，所以这里即使加了也没事，
                # 但为了逻辑清晰，保持这个判断
                x = mu

        self.model.train()

        # 将图像还原回 [0, 1] 范围并 clamp
        x = (x.clamp(-1, 1) + 1) / 2
        return x