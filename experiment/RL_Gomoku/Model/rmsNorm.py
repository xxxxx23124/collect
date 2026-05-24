import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    论文: https://arxiv.org/abs/1910.07467
    Args:
        dim (int): 输入特征的维度。
        eps (float): 为保证数值稳定性加在分母上的一个很小的值。
    """
    def __init__(self, dim, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # 'weight' 是 RMSNorm 的可学习增益参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 归一化后的输出张量。
        """
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
