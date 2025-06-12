import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = (2 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    RMS = sqrt(1/n * Σ(x_i²) + eps)
    y_i = (x_i / RMS) * γ_i
    where γ_i is a learnable parameter.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms * self.weight
        return x.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % 64 == 0, "d_model must be divisible by 64"
        if d_ff is None:
            d_ff = d_model * 8 // 3
        self.d_model = d_model
        self.silu = SiLU()
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.silu(self.W1(x)) * self.W3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.dim = d_k
        # θ = base^(–2i / d)
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )  # (dim / 2, )
        pos = torch.arange(
            max_seq_len, device=device
        )  # (max_seq_len, )
        freqs = einsum(
            pos, inv_freq, "max_seq, dim -> max_seq dim"
        )  # mθ (max_seq dim/2)
        sin = freqs.sin()
        cos = freqs.cos()
        self.register_buffer("sin", sin, persistent=False)  # (max_seq, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # (max_seq, head_dim)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x (..., seq_len, d_k)
        # token_position (batch_size, seq_len)
        sin, cos = self.sin[token_positions], self.cos[token_positions]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        
        return out
