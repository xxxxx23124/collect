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
                 device="cuda"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # 将模型移至设备
        self.model.to(device)

        # ============================================================
        # 1. 预计算扩散过程所需的常数 (Linear Schedule)
        # ============================================================
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        # alpha_hat (cumulative product)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # 预先计算常用的平方根，避免训练时重复计算
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat)

        # 反向过程所需的常数 (用于采样公式)
        # mu = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_hat) * epsilon)
        self.inv_sqrt_alpha = 1.0 / torch.sqrt(self.alpha)
        self.denoise_coeff = self.beta / self.sqrt_one_minus_alpha_hat

        # Sigma (方差)
        self.sigma = torch.sqrt(self.beta)

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
        predicted_noise = self.model(x_t, t)

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
            predicted_noise = self.model(x, t)

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
                sigma_t = self.extract(self.sigma, t, x.shape)
                x = mu + sigma_t * noise
            else:
                x = mu

        self.model.train()

        # 将图像还原回 [0, 1] 范围并 clamp
        x = (x.clamp(-1, 1) + 1) / 2
        return x