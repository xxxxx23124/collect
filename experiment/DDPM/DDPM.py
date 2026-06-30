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

        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

        self.register_buffer("sqrt_alpha_hat", torch.sqrt(alpha_hat))
        self.register_buffer("sqrt_one_minus_alpha_hat", torch.sqrt(1.0 - alpha_hat))

        inv_sqrt_alpha = 1.0 / torch.sqrt(alpha)
        denoise_coeff = beta / torch.sqrt(1.0 - alpha_hat)

        self.register_buffer("inv_sqrt_alpha", inv_sqrt_alpha)
        self.register_buffer("denoise_coeff", denoise_coeff)

        alpha_hat_prev = F.pad(alpha_hat[:-1], (1, 0), value=1.0)
        self.register_buffer("alpha_hat_prev", alpha_hat_prev)

        posterior_variance = beta * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=0.0))

        self.register_buffer("sigma_fl", torch.sqrt(beta))
        self.register_buffer("sigma_fs", torch.sqrt(posterior_variance.clamp(min=0.0)))

    @property
    def device(self):
        return self.beta.device

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        # TimeMLP 使用连续时间输入，这里统一映射到 [0, 1]。
        return t.float() / self.timesteps

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_hat_t = self.extract(self.sqrt_alpha_hat, t, x_0.shape)
        sqrt_one_minus_alpha_hat_t = self.extract(self.sqrt_one_minus_alpha_hat, t, x_0.shape)
        x_t = sqrt_alpha_hat_t * x_0 + sqrt_one_minus_alpha_hat_t * noise

        return x_t, noise

    def compute_loss(self, x_0):
        """
        计算 DDPM 的损失
        x_0: 原始图像 (Batch, C, H, W)，假设范围已经归一化到 [-1, 1]
        """
        batch_size = x_0.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        x_t, noise = self.add_noise(x_0, t)
        predicted_noise = self.model(x_t, self.normalize_time(t))

        loss = F.mse_loss(predicted_noise, noise)
        if hasattr(self.model, "get_router_auxiliary_loss"):
            # Router 正则依赖当前 batch 的 routing graph，必须随主 loss 一起反传。
            loss = loss + self.model.get_router_auxiliary_loss()

        return loss

    @torch.no_grad()
    def sample(self, num_samples, img_size, channels=3):
        was_training = self.model.training
        self.model.eval()

        x = torch.randn(num_samples, channels, img_size, img_size, device=self.device)

        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t = (torch.ones(num_samples) * i).long().to(self.device)
            predicted_noise = self.model(x, self.normalize_time(t))

            inv_sqrt_alpha_t = self.extract(self.inv_sqrt_alpha, t, x.shape)
            denoise_coeff_t = self.extract(self.denoise_coeff, t, x.shape)
            mu = inv_sqrt_alpha_t * (x - denoise_coeff_t * predicted_noise)

            if i > 0:
                noise = torch.randn_like(x)
                if self.use_fixed_small:
                    sigma_t = self.extract(self.sigma_fs, t, x.shape)
                else:
                    sigma_t = self.extract(self.sigma_fl, t, x.shape)
                x = mu + sigma_t * noise
            else:
                x = mu

        self.model.train(was_training)

        x = (x.clamp(-1, 1) + 1) / 2
        return x