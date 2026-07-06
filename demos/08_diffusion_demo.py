"""
08 Diffusion 扩散模型 Toy Demo

核心一句话：Diffusion 先把数据逐步加噪成噪声，再学习一步步去噪生成数据。
整体流程：x0 -> 正向加噪 xt -> 训练网络预测噪声 epsilon -> 反向去噪采样。
分层结构：Forward Process、Denoising Network、Reverse Process、Sampling。
关键机制：噪声调度、时间步 t、噪声预测、逐步去噪。
易错点：Diffusion 不是一次生成结果，而是多步采样；训练目标通常是预测噪声。
记忆口诀：先打乱，再复原，一步一步生成。
自测问题：正向扩散和反向扩散分别做什么？模型训练时预测什么？
"""
from __future__ import annotations

import math
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import matplotlib.pyplot as plt
from common import save_fig, print_block


def make_data(n: int) -> torch.Tensor:
    # 二维圆环数据，便于观察从噪声恢复成结构化分布。
    theta = torch.rand(n) * 2 * math.pi
    r = 2.0 + 0.12 * torch.randn(n)
    x = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    return x


class Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, t_norm], dim=1))


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    T = 40
    betas = torch.linspace(1e-4, 0.04, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练一个小噪声预测网络 epsilon_theta(x_t, t)。
    for step in range(260):
        x0 = make_data(256).to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        eps = torch.randn_like(x0)
        ab = alpha_bars[t].view(-1, 1)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        t_norm = (t.float() / (T - 1)).view(-1, 1)
        pred_eps = model(x_t, t_norm)
        loss = ((pred_eps - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # 展示正向加噪
    x0 = make_data(600)
    forward_steps = [0, 10, 25, 39]
    forward_samples = []
    for s in forward_steps:
        eps = torch.randn_like(x0)
        ab = alpha_bars[s]
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        forward_samples.append(x_t.detach().numpy())

    # 反向采样：从纯噪声开始逐步去噪。
    x = torch.randn(600, 2)
    reverse_snapshots = []
    for t_step in reversed(range(T)):
        t_tensor = torch.full((x.shape[0], 1), t_step / (T - 1))
        with torch.no_grad():
            eps_pred = model(x, t_tensor)
        beta = betas[t_step]
        alpha = alphas[t_step]
        alpha_bar = alpha_bars[t_step]
        x = (x - beta / torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha)
        if t_step in [39, 25, 10, 0]:
            reverse_snapshots.append((t_step, x.detach().numpy().copy()))

    print_block("Diffusion Demo", "在 2D 圆环数据上训练一个小 denoiser，演示正向加噪与反向去噪采样。")
    print("T steps:", T)
    print("训练完成：模型学习预测噪声 epsilon。")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, s, arr in zip(axes[0], forward_steps, forward_samples):
        ax.scatter(arr[:, 0], arr[:, 1], s=5, alpha=0.6)
        ax.set_title(f"Forward t={s}")
        ax.axis("equal"); ax.axis("off")
    for ax, (s, arr) in zip(axes[1], reverse_snapshots):
        ax.scatter(arr[:, 0], arr[:, 1], s=5, alpha=0.6)
        ax.set_title(f"Reverse t={s}")
        ax.axis("equal"); ax.axis("off")
    save_fig("08_diffusion_forward_reverse_demo.png")


if __name__ == "__main__":
    main()
