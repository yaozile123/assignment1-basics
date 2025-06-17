import torch
import torch.nn as nn
from einops import einsum, rearrange


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
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


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
        self.register_buffer("sin", sin, persistent=False)  # (max_seq, dim / 2)
        self.register_buffer("cos", cos, persistent=False)  # (max_seq, dim / 2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # x (batch_size, num_heads, seq_len, d_k)
        # token_position (batch_size, seq_len)
        if token_positions.dim() == 1:
            token_positions = token_positions.unsqueeze(0)
        sin, cos = self.sin[token_positions], self.cos[token_positions] # (batch_size, seq_len, dim / 2)
        if x.dim() == 4:
            sin = sin.unsqueeze(1)
            cos = cos.unsqueeze(1)
        x_even, x_odd = x[..., 0::2], x[..., 1::2] # (batch_size, num_heads, seq_len, dim / 2)
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        # we want to build like (x1, x2, x3, x4) so we need to stack and flatten
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2) # (batch_size, seq_len, d_k)
        
        return out


def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum     


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value :torch.Tensor, mask: torch.Tensor):
    d = query.size()[-1]
    dot_product = einsum(query, key, "b ... q d, b ... k d -> b ... q k") # b, s, s
    wei = dot_product.masked_fill(mask == 0, -float('inf'))
    scaled_dot_product = wei / d ** 0.5
    out = softmax(scaled_dot_product, dim=-1)
    out = einsum(out, value, "b ... q k, b ... k d -> b ... q d") # b, s, d
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: int = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Only create RoPE if both max_seq_len and theta are provided
        self.use_rope = max_seq_len is not None and theta is not None
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        seq_len = x.size(-2)
        batch_size = x.size(0)
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "batch seq (num_heads head_dim) -> batch num_heads seq head_dim", num_heads = self.num_heads)
        K = rearrange(K, "batch seq (num_heads head_dim) -> batch num_heads seq head_dim", num_heads = self.num_heads)
        V = rearrange(V, "batch seq (num_heads head_dim) -> batch num_heads seq head_dim", num_heads = self.num_heads)
        
        # Apply RoPE only if it's enabled
        if self.use_rope:
            # If token_positions is None, create default sequential positions
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
            
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, "batch num_heads seq head_dim -> batch seq (num_heads head_dim)")
        return self.output_proj(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, max_seq_len: int = 8192, theta: int = 10000):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, theta: int = 10000):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)