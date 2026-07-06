# 8 种深度神经网络算法原理 Demo 项目

这个项目用于帮助你快速理解、长期记忆，并能在面试中清晰讲解 8 类常见深度学习算法：

1. CNN 卷积神经网络
2. YOLO 目标检测
3. Density Map 人群密度估计
4. RNN / LSTM 循环神经网络
5. Transformer 注意力机制
6. AutoEncoder 自编码器
7. GAN 生成对抗网络
8. Diffusion 扩散模型

每个 Demo 都遵循同一套学习模板：

```text
核心一句话 → 整体流程 → 分层结构 → 关键机制 → 易错点 → 记忆口诀 → 自测问题
```

## 运行方式

```bash
cd deep_learning_8_algorithms_demo
pip install -r requirements.txt
python run_all.py
```

也可以单独运行某一个 Demo：

```bash
python demos/01_cnn_demo.py
python demos/02_yolo_toy_demo.py
python demos/03_density_map_demo.py
python demos/04_rnn_lstm_demo.py
python demos/05_transformer_attention_demo.py
python demos/06_autoencoder_demo.py
python demos/07_gan_demo.py
python demos/08_diffusion_demo.py
```

运行后会在 `outputs/` 目录生成对应算法的可视化图片。

## 项目结构

```text
deep_learning_8_algorithms_demo/
├── README.md
├── requirements.txt
├── run_all.py
├── docs/
│   └── algorithms_interview_notes.md
├── demos/
│   ├── 01_cnn_demo.py
│   ├── 02_yolo_toy_demo.py
│   ├── 03_density_map_demo.py
│   ├── 04_rnn_lstm_demo.py
│   ├── 05_transformer_attention_demo.py
│   ├── 06_autoencoder_demo.py
│   ├── 07_gan_demo.py
│   └── 08_diffusion_demo.py
└── outputs/
```

## 面试使用建议

你不需要把这些算法讲成“算法研究员”。你的定位是 **CUDA 推理 / AI 推理部署 / 音视频 AI 工程化**，所以面试讲法建议是：

> 我理解这些模型的核心原理，更关注如何把模型接入 C++/CUDA/TensorRT/vLLM/音视频系统中，完成推理部署、性能观察和业务联调。

