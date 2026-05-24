import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSwiGLU(nn.Module, ABC):
    """
    SwiGLU 前馈网络基类。

    通用结构：

        gate = gate_proj(x)
        up = up_proj(x)

        hidden = SiLU(gate) * up
        hidden = hidden_dropout(hidden)

        output = down_proj(hidden)
        output = output_dropout(output)

    子类需要实现 _init_sublayers()，用于初始化：

        self.gate_proj
        self.up_proj
        self.down_proj

    Args:
        input_dim:
            输入维度。

        output_dim:
            输出维度。

        up_proj_dim:
            中间升维维度。

        hidden_dropout:
            作用在 SwiGLU 门控后的中间特征上。

        output_dropout:
            作用在 down_proj 输出之后。

        **kwargs:
            传递给子类 _init_sublayers 的额外参数。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        up_proj_dim: int,
        hidden_dropout: float = 0.0,
        output_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        if not 0.0 <= hidden_dropout <= 1.0:
            raise ValueError(
                f"hidden_dropout must be in [0, 1], got {hidden_dropout}"
            )

        if not 0.0 <= output_dropout <= 1.0:
            raise ValueError(
                f"output_dropout must be in [0, 1], got {output_dropout}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_proj_dim = up_proj_dim

        self.hidden_dropout_p = hidden_dropout
        self.output_dropout_p = output_dropout

        self.silu = nn.SiLU()

        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        # 初始化投影层为 None，由子类负责定义
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

        # 调用抽象方法，强制子类实现层的初始化
        self._init_sublayers(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            up_proj_dim=self.up_proj_dim,
            **kwargs,
        )

        # 确保子类已经正确初始化了所有层
        if self.gate_proj is None:
            raise RuntimeError(
                "self.gate_proj must be initialized in _init_sublayers"
            )

        if self.up_proj is None:
            raise RuntimeError(
                "self.up_proj must be initialized in _init_sublayers"
            )

        if self.down_proj is None:
            raise RuntimeError(
                "self.down_proj must be initialized in _init_sublayers"
            )

    @abstractmethod
    def _init_sublayers(self, **kwargs):
        """
        子类必须实现此方法来初始化：

            self.gate_proj
            self.up_proj
            self.down_proj

        最普通的实现例如：

            input_dim = kwargs["input_dim"]
            output_dim = kwargs["output_dim"]
            up_proj_dim = kwargs["up_proj_dim"]

            self.gate_proj = nn.Linear(input_dim, up_proj_dim)
            self.up_proj = nn.Linear(input_dim, up_proj_dim)
            self.down_proj = nn.Linear(up_proj_dim, output_dim)
        """
        raise NotImplementedError(
            "Subclasses must implement the _init_sublayers method."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                输入张量，常见 shape:

                    (B, S, input_dim)

                也可以是：

                    (B, input_dim)

        Returns:
            output:
                shape 最后一维为 output_dim。
        """

        # 1. 门控分支
        gate = self.gate_proj(x)

        # 2. 内容分支
        up_content = self.up_proj(x)

        # 3. SwiGLU 门控
        gated_hidden = self.silu(gate) * up_content

        # 4. 中间层 Dropout
        gated_hidden = self.hidden_dropout(gated_hidden)

        # 5. 降维投影
        output = self.down_proj(gated_hidden)

        # 6. 输出 Dropout
        output = self.output_dropout(output)

        return output



class SwiGLU(BaseSwiGLU):
    """
    标准 SwiGLU FFN。
    """
    def _init_sublayers(self, **kwargs):
        input_dim = kwargs["input_dim"]
        output_dim = kwargs["output_dim"]
        up_proj_dim = kwargs["up_proj_dim"]

        self.gate_proj = nn.Linear(input_dim, up_proj_dim, bias=False)
        self.up_proj = nn.Linear(input_dim, up_proj_dim, bias=False)
        self.down_proj = nn.Linear(up_proj_dim, output_dim, bias=False)
