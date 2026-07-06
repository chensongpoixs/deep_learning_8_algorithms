"""
02 YOLO 目标检测 Toy Demo

核心一句话：YOLO 一次前向推理同时预测目标位置、类别和置信度。
整体流程：图像 -> 网格划分 -> 每个网格预测候选框 -> 置信度筛选 -> NMS 去重。
分层结构：Backbone 提特征，Neck 融合多尺度，Head 输出 box/objectness/class。
关键机制：一阶段检测、多尺度预测、NMS、置信度阈值。
易错点：YOLO 适合稀疏/中等密度目标，极密集人群要结合 Density Map。
记忆口诀：一眼看全图，多尺度找目标，NMS 去重框。
自测问题：NMS 为什么必要？YOLO 为什么适合实时视频？
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import save_fig, print_block


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order) > 0:
        idx = int(order[0])
        keep.append(idx)
        order = np.array([j for j in order[1:] if iou(boxes[idx], boxes[j]) < threshold])
    return keep


def draw_boxes(ax, boxes, scores, keep=None, title=""):
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    ax.set_title(title)
    for x in np.linspace(0, 1, 5):
        ax.axvline(x, color="lightgray", lw=0.8)
        ax.axhline(x, color="lightgray", lw=0.8)
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        selected = keep is None or i in keep
        lw = 2.5 if selected else 1.0
        alpha = 1.0 if selected else 0.25
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=lw, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x1, max(0.02, y1 - 0.02), f"{score:.2f}", fontsize=8, alpha=alpha)
    ax.scatter([0.32, 0.72], [0.35, 0.62], marker="x", s=80)


def main() -> None:
    # 模拟 YOLO Head 输出的候选框：两个真实目标附近各有多个重叠预测框。
    boxes = np.array([
        [0.20, 0.20, 0.45, 0.50],
        [0.22, 0.22, 0.47, 0.52],
        [0.18, 0.18, 0.42, 0.48],
        [0.58, 0.48, 0.88, 0.78],
        [0.60, 0.50, 0.86, 0.76],
        [0.10, 0.70, 0.24, 0.88],  # 低置信度误检
    ], dtype=np.float32)
    scores = np.array([0.92, 0.86, 0.60, 0.90, 0.78, 0.23], dtype=np.float32)
    mask = scores > 0.5
    boxes_filtered = boxes[mask]
    scores_filtered = scores[mask]
    keep = nms(boxes_filtered, scores_filtered, threshold=0.4)

    print_block("YOLO Toy Demo", "模拟 YOLO 候选框输出：先按置信度过滤，再用 NMS 删除重复框。")
    print("候选框数量:", len(boxes))
    print("过滤后数量:", len(boxes_filtered))
    print("NMS 保留索引:", keep)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw_boxes(axes[0], boxes, scores, title="Before Filtering + NMS")
    draw_boxes(axes[1], boxes_filtered, scores_filtered, keep=keep, title="After Confidence Filter + NMS")
    save_fig("02_yolo_grid_nms_demo.png")


if __name__ == "__main__":
    main()
