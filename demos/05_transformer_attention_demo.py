"""
05 Transformer Self-Attention Demo

核心一句话：Transformer 用 Self-Attention 让每个 token 直接关注其他 token。
整体流程：Token -> Embedding -> Q/K/V -> Attention 权重 -> 加权 V -> 上下文表示。
分层结构：Embedding、Position Encoding、Multi-Head Attention、FFN、Residual、LayerNorm。
关键机制：Q/K/V、Softmax 权重、多头注意力、位置编码。
易错点：Attention 不是简单查表，QK 得权重，再对 V 加权求和。
记忆口诀：Q 问问题，K 做匹配，V 给答案。
自测问题：为什么要除以 sqrt(d_k)？Multi-Head 有什么意义？
"""
from __future__ import annotations

import math
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from common import save_fig, print_block


def main() -> None:
    torch.manual_seed(0)
    tokens = ["UAV", "find", "square", "crowd", "dense", "alarm"]
    n, d = len(tokens), 16
    x = torch.randn(n, d)

    Wq = torch.randn(d, d) / math.sqrt(d)
    Wk = torch.randn(d, d) / math.sqrt(d)
    Wv = torch.randn(d, d) / math.sqrt(d)
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / math.sqrt(d)
    attn = torch.softmax(scores, dim=-1)
    context = attn @ v

    print_block("Transformer Attention Demo", "手写 Q/K/V 和 Self-Attention，观察每个 token 关注其他 token 的权重矩阵。")
    print("tokens:", tokens)
    print("context shape:", tuple(context.shape))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn.detach().numpy(), cmap="viridis")
    ax.set_title("Self-Attention Weight Matrix")
    ax.set_xticks(range(n), tokens, rotation=45, ha="right")
    ax.set_yticks(range(n), tokens)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{attn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax)
    save_fig("05_transformer_attention_matrix.png")


if __name__ == "__main__":
    main()
