"""
03 Density Map 人群密度估计 Demo

核心一句话：密度图不逐个框人，而是预测人群分布，对密度图求和得到人数。
整体流程：Head Point -> Gaussian Kernel -> Density Map -> 积分人数 -> 热力图。
分层结构：输入图像、点标注、密度图标签、CNN 预测、区域积分。
关键机制：Head Point 标注、Gaussian Density、Density Sum、多尺度感受野。
易错点：密集人群估计不是目标检测，不能只靠 YOLO 框人。
记忆口诀：点变热图，热图求和，密集不框人。
自测问题：为什么密集场景不用 YOLO 直接数人？Density Map 怎么得到人数？
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from common import save_fig, print_block


def add_gaussian(density: np.ndarray, point: tuple[int, int], sigma: float = 2.0) -> None:
    h, w = density.shape
    y0, x0 = point
    radius = int(3 * sigma)
    y_min, y_max = max(0, y0 - radius), min(h, y0 + radius + 1)
    x_min, x_max = max(0, x0 - radius), min(w, x0 + radius + 1)
    ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
    g = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))
    g = g / (g.sum() + 1e-8)  # 每个人贡献总和约等于 1
    density[y_min:y_max, x_min:x_max] += g


def main() -> None:
    h, w = 96, 128
    rng = np.random.default_rng(7)
    sparse_points = np.array([[20, 22], [28, 80], [65, 100]])
    cluster = rng.normal(loc=[55, 55], scale=[9, 13], size=(38, 2)).astype(int)
    cluster[:, 0] = np.clip(cluster[:, 0], 0, h - 1)
    cluster[:, 1] = np.clip(cluster[:, 1], 0, w - 1)
    points = np.vstack([sparse_points, cluster])

    density = np.zeros((h, w), dtype=np.float32)
    for y, x in points:
        add_gaussian(density, (int(y), int(x)), sigma=2.3)

    estimated_count = density.sum()
    true_count = len(points)

    print_block("Density Map Demo", "每个人头点扩散成一个总和为 1 的高斯分布，整张密度图求和即人数估计。")
    print(f"真实人数: {true_count}")
    print(f"密度图积分估计人数: {estimated_count:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    canvas = np.zeros((h, w))
    axes[0].imshow(canvas, cmap="gray")
    axes[0].scatter(points[:, 1], points[:, 0], s=14, c="white", edgecolors="black")
    axes[0].set_title("Head Point Annotations")
    axes[0].axis("off")

    im = axes[1].imshow(density, cmap="hot")
    axes[1].set_title("Gaussian Density Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(density, cmap="hot")
    axes[2].contour(density, levels=6, colors="white", linewidths=0.5)
    axes[2].set_title(f"Integral Count ≈ {estimated_count:.1f}")
    axes[2].axis("off")
    save_fig("03_density_map_counting_demo.png")


if __name__ == "__main__":
    main()
