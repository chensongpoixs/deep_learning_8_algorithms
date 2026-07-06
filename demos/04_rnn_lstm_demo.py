"""
04 RNN / LSTM 序列建模 Demo

核心一句话：RNN 用隐藏状态记住过去，LSTM 用门控机制解决长期记忆问题。
整体流程：序列输入 -> 时间步展开 -> 更新隐藏状态 -> 输出预测。
分层结构：RNN Cell；LSTM 的遗忘门、输入门、候选记忆、输出门、Cell State。
关键机制：隐藏状态传递、门控记忆、长期依赖。
易错点：普通 RNN 容易梯度消失，LSTM 通过 Cell State 保留长期信息。
记忆口诀：RNN 记过去，LSTM 加门控。
自测问题：LSTM 三个门分别干什么？为什么普通 RNN 不擅长长序列？
"""
from __future__ import annotations

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import matplotlib.pyplot as plt
from common import save_fig, print_block


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyRNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out)


class TinyLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out)


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, steps: int = 60) -> list[float]:
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(steps):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    set_seed()
    t = torch.linspace(0, 8 * np.pi, 120)
    series = torch.sin(t) + 0.2 * torch.sin(3 * t)
    x = series[:-1].view(1, -1, 1)
    y = series[1:].view(1, -1, 1)

    rnn = TinyRNN()
    lstm = TinyLSTM()
    rnn_loss = train(rnn, x, y)
    lstm_loss = train(lstm, x, y)

    with torch.no_grad():
        rnn_pred = rnn(x).view(-1).numpy()
        lstm_pred = lstm(x).view(-1).numpy()

    print_block("RNN / LSTM Demo", "用 RNN 和 LSTM 学习一个简单正弦序列的下一步预测。")
    print(f"RNN final loss : {rnn_loss[-1]:.6f}")
    print(f"LSTM final loss: {lstm_loss[-1]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(y.view(-1).numpy(), label="target")
    axes[0].plot(rnn_pred, label="RNN pred", alpha=0.8)
    axes[0].plot(lstm_pred, label="LSTM pred", alpha=0.8)
    axes[0].set_title("Sequence Prediction")
    axes[0].legend()

    axes[1].plot(rnn_loss, label="RNN loss")
    axes[1].plot(lstm_loss, label="LSTM loss")
    axes[1].set_title("Training Loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    save_fig("04_rnn_lstm_sequence_demo.png")


if __name__ == "__main__":
    main()
