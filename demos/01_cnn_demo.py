"""
01 CNN 卷积神经网络 Demo

核心一句话：CNN 用卷积核扫描图像，从局部纹理逐层抽象到高级语义。
整体流程：输入图像 -> 卷积 -> ReLU -> 池化 -> 全连接/分类。
分层结构：浅层边缘纹理，中层局部结构，深层目标语义。
关键机制：局部感受野、权重共享、ReLU 非线性、池化降维。
易错点：CNN 不是只用于分类，YOLO、密度估计、视频 AI 都可基于 CNN Backbone。
记忆口诀：卷积看局部，池化降尺寸，层层抽语义。
自测问题：卷积为什么比全连接更适合图像？池化为什么能降计算？
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from common import save_fig, print_block


def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    out = np.zeros((oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def max_pool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    h, w = x.shape
    out = np.zeros((h // size, w // size), dtype=np.float32)
    for i in range(0, h - size + 1, size):
        for j in range(0, w - size + 1, size):
            out[i // size, j // size] = np.max(x[i:i + size, j:j + size])
    return out


def main() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    image[4:12, 5:11] = 1.0
    image[7:9, 2:14] = 0.7

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    conv = conv2d(image, sobel_x)
    act = relu(conv)
    pooled = max_pool2d(act, size=2)

    print_block("CNN Demo", "输入图像经过 Sobel 卷积提取竖直边缘，再经过 ReLU 和 MaxPooling。")
    print("image shape:", image.shape)
    print("conv shape:", conv.shape)
    print("pooled shape:", pooled.shape)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["Input Image", "Conv: Edge Feature", "ReLU", "MaxPool"]
    arrays = [image, conv, act, pooled]
    for ax, title, arr in zip(axes, titles, arrays):
        ax.imshow(arr, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    save_fig("01_cnn_feature_pipeline.png")


if __name__ == "__main__":
    main()
