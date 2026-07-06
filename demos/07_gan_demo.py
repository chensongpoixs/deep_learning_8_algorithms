"""
07 GAN 生成对抗网络 Demo

核心一句话：GAN 让生成器造假、判别器鉴假，通过对抗训练生成逼真数据。
整体流程：噪声 z -> Generator -> 假样本 -> Discriminator -> 真/假判断 -> 对抗优化。
分层结构：Generator、Discriminator、Adversarial Loss。
关键机制：真假样本分布逼近、D 给 G 反馈、min-max 博弈。
易错点：GAN 训练不稳定，可能模式崩溃，不是训练越久一定越好。
记忆口诀：G 造假，D 鉴假，对抗变真。
自测问题：G 和 D 分别优化什么？什么是模式崩溃？
"""
from __future__ import annotations

import torch
torch.set_num_threads(1)
import torch.nn as nn
import matplotlib.pyplot as plt
from common import save_fig, print_block


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_real(batch: int) -> torch.Tensor:
    # 真实数据：一维高斯分布，均值 4，方差 1
    return torch.randn(batch, 1) + 4.0


def main() -> None:
    torch.manual_seed(0)
    g, d = Generator(), Discriminator()
    opt_g = torch.optim.Adam(g.parameters(), lr=0.003)
    opt_d = torch.optim.Adam(d.parameters(), lr=0.003)
    bce = nn.BCELoss()
    history = []

    for step in range(260):
        real = sample_real(128)
        z = torch.randn(128, 1)
        fake = g(z).detach()

        d_real = d(real)
        d_fake = d(fake)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(128, 1)
        fake = g(z)
        d_fake_for_g = d(fake)
        loss_g = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % 100 == 0:
            with torch.no_grad():
                gen_mean = g(torch.randn(1000, 1)).mean().item()
            history.append((step, loss_d.item(), loss_g.item(), gen_mean))

    with torch.no_grad():
        real = sample_real(2000).numpy().ravel()
        fake = g(torch.randn(2000, 1)).numpy().ravel()

    print_block("GAN Demo", "用极小 GAN 学习一维真实分布 N(4,1)，观察生成分布向真实分布靠近。")
    print("history: step, loss_d, loss_g, generated_mean")
    for row in history[-5:]:
        print(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(real, bins=50, alpha=0.6, density=True, label="real")
    axes[0].hist(fake, bins=50, alpha=0.6, density=True, label="generated")
    axes[0].set_title("Real vs Generated Distribution")
    axes[0].legend()

    steps = [h[0] for h in history]
    gen_means = [h[3] for h in history]
    axes[1].plot(steps, gen_means, marker="o")
    axes[1].axhline(4.0, linestyle="--", label="real mean=4")
    axes[1].set_title("Generated Mean Approaches Real Mean")
    axes[1].legend()
    save_fig("07_gan_1d_distribution_demo.png")


if __name__ == "__main__":
    main()
