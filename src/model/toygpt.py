from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.rich import tqdm


class MultiHeadCasualSelfAttention(nn.Module):

    def __init__(self, seq_len: int, embed_size: int, hidden_size: int,
                 n_head: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.n_head = n_head
        self.hidden_size = hidden_size

        self.multiqkv = nn.Linear(self.embed_size,
                                  3 * self.n_head * self.hidden_size)
        self.out_proj = nn.Linear(self.n_head * self.hidden_size,
                                  self.embed_size)

        self.dropout = nn.Dropout(dropout)

        tril_mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        self.register_buffer("mask",
                             tril_mask.view(1, 1, self.seq_len, self.seq_len))

        inv_scale = 1.0 / (self.hidden_size**0.5)
        self.register_buffer("inv_scale", torch.tensor(inv_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B: batch size, S: sequence length, E: embed size, H: hidden size, N: number of heads
        B, S, E = x.shape
        N, H = self.n_head, self.hidden_size
        # (B, S, E) -> (B, S, 3 * N * H)
        multiqkv: torch.Tensor = self.multiqkv(x)

        # (B, S, 3 * N * H) -> 3 * (B, S, N * H)
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        q, k, v = multiqkv.chunk(3, dim=-1)

        # (B, S, N * H) -> (B, S, N, H)
        q = q.view(B, S, N, H)
        k = k.view(B, S, N, H)
        v = v.view(B, S, N, H)

        # (B, S, N, H) -> (B, N, S, H)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (Q * K^T) / scale: (B, N, S, H) @ (B, N, H, S) -> (B, N, S, S)
        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * self.inv_scale

        # mask
        attn = attn.masked_fill(self.mask[:, :, :S, :S] == 0, float('-inf'))

        # softmax & dropout
        attn = self.dropout(attn.softmax(dim=-1))

        # softmax * V: (B, N, S, S) @ (B, N, S, H) -> (B, N, S, H)
        attn = attn @ v

        # put head back to together
        # (B, N, S, H) -> (B, S, N, H) -> (B, S, N * H)
        attn = attn.transpose(1, 2).contiguous().view(B, S, N * H)

        # output projection & dropout
        # (B, S, N * H) -> (B, S, E)
        attn = self.dropout(self.out_proj(attn))

        return attn


class FeedForward(nn.Module):

    def __init__(self, embed_size: int, expansion_factor: int,
                 dropout: float) -> None:
        super().__init__()
        self.up_scale = nn.Linear(embed_size, embed_size * expansion_factor)
        self.gelu = nn.GELU()
        self.down_proj = nn.Linear(embed_size * expansion_factor, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.up_scale(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, seq_len: int,
                 embed_size: int, hidden_size: int,
                 n_head: int, expansion_factor: int,
                 dropout: float) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.attention = MultiHeadCasualSelfAttention(seq_len, embed_size,
                                                 hidden_size, n_head, dropout)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.feedforward = FeedForward(embed_size, expansion_factor, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layernorm1(x)
        x2 = self.attention(x1)
        x3 = x2 + x # residual connection
        x4 = self.layernorm2(x3)
        x5 = self.feedforward(x4)
        x6 = x5 + x3 # residual connection
        return x6

class toyGPT(nn.Module):

    def __init__(self, n_block: int, seq_len: int, embed_size: int,
                 hidden_size: int, n_head: int, expansion_factor: int,
                 dropout: float, vacab_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(vacab_size, embed_size)
        self.position_embed = nn.Embedding(seq_len, embed_size)

        self.transformer = nn.Sequential(*[
            Block(seq_len, embed_size, hidden_size, n_head, expansion_factor,
                  dropout) for _ in range(n_block)
        ])

        self.layernorm = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vacab_size)
        self.lm_head.weight = self.token_embed.weight
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor,
                targets: Union[torch.Tensor, None] = None) -> torch.Tensor:
        device = inputs.device
        dtype = inputs.dtype
        _, S = inputs.shape
        assert S <= self.seq_len, \
            "input sequence length is longer than model sequence length"
        position = torch.arange(S, dtype=dtype, device=device)

        # Embedding
        x = self.token_embed(inputs) + self.position_embed(position)
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Final Normalization
        x = self.layernorm(x)

        # Output
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def inference(self, cond_token: torch.Tensor, max_new_tokens: int = 200) -> torch.Tensor:
        outputs = cond_token
        for _ in tqdm(range(max_new_tokens)):
            cond_token = outputs[:, -self.seq_len:]
            logits, _ = self.forward(cond_token)
            logits = logits[:, -1, :]
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            outputs = torch.cat([outputs, next_token], dim=-1)

        return outputs

