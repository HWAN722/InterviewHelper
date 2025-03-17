import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, dim, base=10000, max_seq_len=256):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float()/dim))

        position = torch.arange(max_seq_len)
        sinusoid = position.unsqueeze(1) * inv_freq.unsqueeze(0)
        # equals to sinusoid = torch.einsum('i,j->ij', position, inv_freq)

        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        self.register_buffer("emb", emb) # move to buffer, no need gradient

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device)

        rot_mat = self.emb[position_ids]  # [batch_size, seq_len, dim]

        x = x.reshape(*x.shape[:-1], -1, 2)  # [..., dim/2, 2]

        # x * (cosθ + i sinθ)
        # equals to: real*cosθ - imag*sinθ, real*sinθ + imag*cosθ
        output = torch.stack([
            x[..., 0] * rot_mat[..., 0] - x[..., 1] * rot_mat[..., 1],
            x[..., 0] * rot_mat[..., 1] + x[..., 1] * rot_mat[..., 0]
        ], dim=-1).reshape_as(x)
        return output


dim = 512
seq_len = 256
batch_size = 4

rope = RoPE(dim)

x = torch.randn(batch_size, seq_len, dim)

output = rope(x)

print(output.shape)  # torch.Size([4, 256, 256, 2])

