# %%
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pos = torch.arange(max_len)[:, None]

        i = torch.arange(0, d_model, 2)

        freq = 1 / (10000 ** (i / d_model))

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)

        self.register_buffer("pe", pe[None])

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=5000, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1 / self.d_k**0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))

        self.register_buffer("mask", mask[None, None])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=False):

        B, T, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.reshape(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores * self.scale

        if causal:
            mask = self.mask[:, :, :T, :T]

            scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        heads = attention @ V

        heads = heads.transpose(1, 2).reshape(B, T, D)

        return self.W_o(heads)


# %%


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# %%


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, max_len=max_len
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal=False):

        attn_out = self.attention(x, causal=causal)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# %%


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_ff, dropout=dropout, max_len=max_len
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        self.output_layer.weight = self.token_embedding.weight

    def forward(self, x):

        x = self.token_embedding(x)
        x = x * self.d_model**0.5
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, causal=True)

        logits = self.output_layer(x)

        return logits


# %%
