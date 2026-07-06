"""
06 AutoEncoder 自编码器 Demo

核心一句话：AutoEncoder 把输入压缩成低维表示，再尽量重建原始输入。
整体流程：x -> Encoder -> z -> Decoder -> x' -> 重建误差。
分层结构：Encoder、Latent Code、Decoder、Reconstruction Loss。
关键机制：信息瓶颈、降维、重建误差、异常检测。
易错点：AutoEncoder 不是分类器，目标是重建输入。
记忆口诀：先压缩，再还原，误差看异常。
自测问题：AutoEncoder 为什么能用于异常检测？隐变量 z 的作用是什么？
"""
from __future__ import annotations

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import matplotlib.pyplot as plt
from common import save_fig, print_block


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    t = np.linspace(0, 2 * np.pi, 240)
    # 构造一个接近 1D 流形的 2D 曲线，便于 AutoEncoder 压缩到 1D。
    x_np = np.stack([np.cos(t), np.sin(t) * 0.4], axis=1).astype("float32")
    x_np += 0.03 * np.random.randn(*x_np.shape).astype("float32")
    x = torch.tensor(x_np)

    model = AutoEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(180):
        recon, _ = model(x)
        loss = ((recon - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        recon, z = model(x)
    print_block("AutoEncoder Demo", "将二维曲线压缩成一维隐变量 z，再从 z 重建二维输入。")
    print("final reconstruction MSE:", losses[-1])
    print("latent z shape:", tuple(z.shape))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(x_np[:, 0], x_np[:, 1], s=12, label="input")
    axes[0].set_title("Input 2D Data")
    axes[0].axis("equal")

    axes[1].scatter(recon[:, 0].numpy(), recon[:, 1].numpy(), s=12, label="recon")
    axes[1].set_title("Reconstructed Data")
    axes[1].axis("equal")

    axes[2].plot(losses)
    axes[2].set_title("Reconstruction Loss")
    axes[2].set_yscale("log")
    save_fig("06_autoencoder_reconstruction_demo.png")


if __name__ == "__main__":
    main()
