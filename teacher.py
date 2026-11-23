"""Train-only Transformer teacher with KibbleLM's vocabulary."""

import torch
from torch import nn
import torch.nn.functional as F

from model import VOCAB


class TeacherBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.channel_norm = nn.RMSNorm(width)
        self.channels = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"),
                                      nn.Linear(4 * width, width))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.attention_norm(x)).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        mixed = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        x = x + self.attention_out(mixed.transpose(1, 2).reshape(batch, length, width))
        return x + self.channels(self.channel_norm(x))


class TeacherLM(nn.Module):
    context = 128

    def __init__(self):
        super().__init__()
        width = 192
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Parameter(torch.empty(self.context, width))
        self.blocks = nn.Sequential(*(TeacherBlock(width, 4) for _ in range(4)))
        self.norm = nn.RMSNorm(width)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def logits(self, tokens, suffix):
        x = self.blocks(self.embedding(tokens) + self.position[:tokens.shape[1]])
        return F.linear(self.norm(x[:, -suffix:]), self.embedding.weight)

    def forward(self, tokens):
        return self.logits(tokens, 1)[:, 0]
