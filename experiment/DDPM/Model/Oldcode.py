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

        self.cln = AdaCLN(in_channels, time_emb_dim)
        self.act = nn.GELU()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = make_conv(in_channels, out_channels + dense_inc, 3,1,1,groups)

    def forward(self, x, time_emb):
        def main_path(x_in: torch.Tensor, time_emb_in: torch.Tensor):
            x_in = self.cln(x_in, time_emb_in)
            x_in = self.act(x_in)
            x_in = self.upsample(x_in)
            x_in = self.conv(x_in, time_emb_in)
            return x_in

        return main_path(x, time_emb)
"""

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