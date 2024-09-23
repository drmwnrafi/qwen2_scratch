import torch
import torch.nn as nn
import torch.nn.functional as F

RED, END = '\033[91m', '\033[0m'

class RMSNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super(RMSNorm, self).__init__()
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.embedding_size))

    def forward(self, x: torch.Tensor):
        denom = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((self.weight * x) / denom).type_as(x)

class RoPE(nn.Module):
    def __init__(self, head_dim:int, config: ModelConfig):
        super(RoPE, self).__init__()
        assert head_dim % 2 == 0, "Dimension must be even (divisible by 2)"
        theta = 1 / config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("theta", theta)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        device = x.device
        self.theta = self.theta.to(device)
        m = torch.arange(x.size(1), device=device)
        frequencies = torch.outer(m, self.theta)
        frq_complex = torch.exp(1j * frequencies)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        freq = frq_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freq
        x_rope = torch.stack((x_rotated.real, x_rotated.imag), dim=-1)
        x_rope = torch.flatten(x_rope, start_dim=-2)
        return x_rope.type_as(x)

def ScaleDotProduct(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -torch.inf)
    scores = F.softmax(scores, dim=-1)
    scores = torch.matmul(scores, V)
    return scores

class MultiQuery(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MultiQuery, self).__init__()
        assert config.embedding_size % config.n_q_heads == 0, (
            f"{RED}Embedding size ({config.embedding_size}) is not divisible by query heads ({config.n_q_heads}).{END}"
        )
        self.n_heads = config.n_q_heads
        self.head_dim = config.embedding_size // self.n_heads
        self.q_proj = nn.ModuleList([nn.Linear(config.embedding_size, self.head_dim, bias=config.bias) for _ in range(self.n_heads)])
        self.k_proj = nn.Linear(config.embedding_size, self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.embedding_size, self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.embedding_size, config.embedding_size, bias=config.bias)
        self.rope = RoPE(self.head_dim, config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, _ = x.size()
        K = self.k_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
        K = self.rope(K)
        output = []
        for i in range(self.n_heads):
            Q = self.q_proj[i](x).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)
            Q = self.rope(Q)
            output.append(ScaleDotProduct(Q, K, V, mask))
        output = torch.cat(output, dim=-1).contiguous().view(batch, seq_len, -1)
        return self.out_proj(output)

class GroupedQuery(nn.Module):
    def __init__(self, config: ModelConfig):
        super(GroupedQuery, self).__init__()
        assert config.n_q_heads % config.n_kv_heads == 0, (
            f"{RED}Query heads ({config.n_q_heads}) is not divisible by key and value heads ({config.n_kv_heads}), "
            f"which will result in a non-integer number of groups. {END}"
        )

        self.n_groups = config.n_q_heads // config.n_kv_heads
        self.groups = nn.ModuleList([
            MultiQuery(config) for _ in range(config.n_kv_heads)
        ])
        self.out_proj = nn.Linear(config.embedding_size * config.n_kv_heads, config.embedding_size, bias=config.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, _ = x.size()
        output = torch.cat([mqa(x, mask) for mqa in self.groups], dim=-1)
        return self.out_proj(output)

class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super(SwiGLU, self).__init__()
        self.beta = config.swiglu_beta

    def sigmoid(self, x: torch.Tensor):
        return 1 / (1 + torch.exp(-x * self.beta))

    def forward(self, x: torch.Tensor):
        return x * self.sigmoid(x)

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.embedding_size, config.intermediate_dim * 2, bias=config.bias)
        self.fc2 = nn.Linear(config.intermediate_dim, config.embedding_size, bias=config.bias)
        self.swiGLU = SwiGLU(config)

    def forward(self, x: torch.Tensor):
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x2 = self.swiGLU(x2)
        x = x1 * x2
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Decoder, self).__init__()
        self.RMSNorm = RMSNorm(config)
        self.GQA = GroupedQuery(config)
        self.FFN = FeedForward(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x1 = x
        x = self.RMSNorm(x)
        x = self.GQA(x, mask)

        x2 = x1 + x
        x = self.RMSNorm(x2)
        x = self.FFN(x)

        return x2 + x

class Qwen2(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Qwen2, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.layers = nn.ModuleList([
            Decoder(config) for _ in range(config.n_layers)
        ])
        self.RMSNorm = RMSNorm(config)
        self.fc_out = nn.Linear(config.embedding_size, config.vocab_size, bias=config.bias)

    def forward(self, tokens, mask: torch.Tensor = None):
        x = self.token_embedding(tokens)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.RMSNorm(x)
        return self.fc_out(x)
