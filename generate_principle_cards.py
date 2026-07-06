from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'outputs' / '中文详细原理图'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

ALGORITHMS = [
    {
        'filename': '01_CNN_中文详细原理图.png',
        'title': 'CNN 卷积神经网络',
        'core': '用卷积核在图像上滑动提取局部特征，再逐层组合成边缘→纹理→形状→语义，从而完成分类或特征提取。',
        'flow': ['输入图像', '卷积 Conv', '激活 ReLU', '池化 Pooling', '多层特征堆叠', '全连接/分类头', '输出类别或特征'],
        'layers': ['输入层：接收像素矩阵', '卷积层：提取边缘、纹理、局部模式', '激活层：增加非线性表达能力', '池化层：降采样，减少计算量', '深层卷积：组合成更高级语义', '输出层：分类 / 回归 / 特征向量'],
        'mech': ['局部感受野：一次只看局部区域', '权重共享：同一个卷积核扫描整张图，参数更少', '多通道卷积：同时提取多种特征', '层次化特征：浅层学边缘，深层学语义'],
        'pitfalls': ['卷积核、步长、padding 搞混', '池化不是必须；现代网络有时用 stride conv 代替', '卷积输出尺寸不会算', '以为 CNN 只能做分类，其实检测/分割都用它做特征提取'],
        'mnemonic': '局部看、重复扫、层层抽、最后判。',
        'quiz': ['为什么 CNN 参数比全连接网络少？', '卷积层和池化层分别做什么？', '为什么浅层特征和深层特征不同？'],
    },
    {
        'filename': '02_YOLO_中文详细原理图.png',
        'title': 'YOLO 目标检测',
        'core': '把目标检测看成一次前向推理中的回归与分类问题，直接输出框的位置、类别和置信度。',
        'flow': ['输入图像', 'Backbone 提特征', 'Neck 多尺度融合', 'Head 预测框/类别/置信度', '阈值过滤', 'NMS 去重', '得到最终目标框'],
        'layers': ['Backbone：提取高低层视觉特征', 'Neck：FPN/PAN 融合多尺度信息', 'Head：预测 box、objectness、class score', '后处理：阈值筛选 + NMS'],
        'mech': ['一阶段检测：不先生成候选框，直接预测结果', '多尺度检测：兼顾大目标和小目标', 'Objectness：判断该位置是否有目标', 'NMS：去掉重叠重复框'],
        'pitfalls': ['把置信度、类别概率、最终分数混为一谈', '不会解释 NMS 为什么需要', '只会说快，不会说为什么快', '忽略小目标、密集目标时检测效果会下降'],
        'mnemonic': '一眼看全图，三件事同出：框、类、分；最后 NMS 去重复。',
        'quiz': ['YOLO 为什么属于一阶段检测？', 'Backbone、Neck、Head 分别做什么？', 'NMS 解决的核心问题是什么？'],
    },
    {
        'filename': '03_DensityMap_中文详细原理图.png',
        'title': 'Density Map 人群密度估计',
        'core': '不逐个框出每个人，而是预测一张密度图；对密度图积分即可得到人数估计，适合密集人群。',
        'flow': ['输入无人机图像', 'Head Point 标注', '高斯核生成真实密度图', 'CNN/多尺度网络训练', '输出预测密度图', '密度图求和', '得到人数与热力图'],
        'layers': ['标注层：给每个人头打点', '标签层：把点扩成 Gaussian Density Map', '特征提取层：提取空间特征', '预测层：输出同尺寸或降采样密度图', '统计层：对像素积分得到人数'],
        'mech': ['点标注比框标注更适合密集人群', '高斯核让离散点变成可学习的连续分布', '积分等于总人数，这是核心数学直觉', '热力图便于展示聚集区域'],
        'pitfalls': ['把密度图方法误认为目标检测', '不会解释为什么密集场景不适合纯 YOLO', '忘记人数来自密度图积分', '忽略尺度变化和透视影响'],
        'mnemonic': '不框人、看热度；热度一积分，就是总人数。',
        'quiz': ['为什么密集人群更适合 Density Map？', 'Ground Truth Density Map 是怎么生成的？', '人数为什么等于密度图求和？'],
    },
    {
        'filename': '04_RNN_LSTM_中文详细原理图.png',
        'title': 'RNN / LSTM 序列建模',
        'core': '序列中的当前输出不仅依赖当前输入，还依赖过去状态；LSTM 通过门控机制解决长依赖难记的问题。',
        'flow': ['序列输入 x1,x2,x3...', 'RNN/LSTM 逐步处理', '隐藏状态向后传递', '输出每一步结果或最终结果'],
        'layers': ['输入层：接收时间序列', '循环单元：共享参数重复计算', '隐藏状态：携带历史信息', '输出层：做分类/预测', 'LSTM 额外包含遗忘门、输入门、输出门、Cell State'],
        'mech': ['RNN：h_t = f(x_t, h_{t-1})，天然适合时序', 'LSTM：门控决定忘掉什么、记住什么、输出什么', 'Cell State：长程记忆通道', '共享参数：每个时间步用同一单元'],
        'pitfalls': ['把 hidden state 和 cell state 搞混', '不知道普通 RNN 为什么梯度消失', '不会解释三种门的作用', '以为序列模型只能做 NLP，其实语音/时序预测也常用'],
        'mnemonic': 'RNN 会传旧状态；LSTM 三道门：忘、写、出。',
        'quiz': ['普通 RNN 的核心缺点是什么？', 'LSTM 的遗忘门在做什么？', '为什么 LSTM 比 RNN 更擅长长序列？'],
    },
    {
        'filename': '05_Transformer_中文详细原理图.png',
        'title': 'Transformer / Self-Attention',
        'core': '让每个 token 都能直接关注序列中的其他 token，用注意力机制替代逐步递归，从而高效建模全局关系。',
        'flow': ['文本输入', 'Tokenizer', 'Embedding + Position Encoding', '多头自注意力', '前馈网络 FFN', '残差 + LayerNorm', '堆叠多层', '输出语义表示或下一个 token'],
        'layers': ['输入层：token ids', '嵌入层：变成向量', '位置编码：补充顺序信息', 'Self-Attention：建模 token 间关系', 'FFN：逐位置非线性变换', '残差和归一化：稳定训练'],
        'mech': ['Q、K、V：Query 问别人，Key 表身份，Value 给信息', 'Attention = softmax(QK^T/sqrt(d))V', 'Multi-Head：多个子空间并行看关系', '并行计算：比 RNN 更适合长序列和大模型'],
        'pitfalls': ['分不清 Q/K/V 含义', '只会背公式，不会解释注意力直觉', '忘记位置编码的重要性', '把 Encoder-only、Decoder-only、Encoder-Decoder 混淆'],
        'mnemonic': '先编码位置，再问全局关系；Q 去问、K 来配、V 给答案。',
        'quiz': ['Self-Attention 为什么能建模长距离依赖？', 'Q、K、V 分别代表什么？', '为什么需要位置编码和多头注意力？'],
    },
    {
        'filename': '06_AutoEncoder_中文详细原理图.png',
        'title': 'AutoEncoder 自编码器',
        'core': '先把输入压缩成低维隐变量，再重建回来；如果能重建好，说明学到了数据的核心表示。',
        'flow': ['输入数据 X', 'Encoder 编码压缩', '隐变量 Z', 'Decoder 解码重建', "输出 X'", '计算重建误差'],
        'layers': ['输入层：原始样本', 'Encoder：逐步压缩维度', 'Bottleneck：最关键的低维表示', 'Decoder：从低维表示恢复原输入', '损失层：MSE/BCE 等重建误差'],
        'mech': ['压缩迫使模型保留最有信息量的特征', '隐变量 Z 是数据的低维表达', '重建误差越小，说明提取的表示越有效', '异常检测里，异常样本通常重建差'],
        'pitfalls': ['把自编码器误当成分类器', '不知道它为什么能做异常检测', '忽略瓶颈层的重要性', '只会讲结构，不会讲训练目标是重建误差'],
        'mnemonic': '先压缩，再还原；还原越准，特征越纯。',
        'quiz': ['AutoEncoder 的训练目标是什么？', '为什么它可以用于异常检测？', 'Bottleneck 层的作用是什么？'],
    },
    {
        'filename': '07_GAN_中文详细原理图.png',
        'title': 'GAN 生成对抗网络',
        'core': '生成器负责造假，判别器负责识假；在对抗博弈中，生成器学会生成越来越真实的数据。',
        'flow': ['随机噪声 z', '生成器 G 产生假样本', '真实样本 + 假样本送入判别器 D', 'D 判真假', '反向更新 G 和 D', '循环对抗训练'],
        'layers': ['Generator：把噪声映射成样本', 'Discriminator：判断样本真伪', '损失函数：一个想骗过，一个想识别出', '训练循环：交替优化 G 和 D'],
        'mech': ['最小最大博弈：G 想最小化被识破概率，D 想最大化识别能力', '噪声输入让 G 能采样不同结果', 'D 提供学习信号，逼着 G 逼近真实分布', '收敛后假样本分布接近真实分布'],
        'pitfalls': ['不知道为什么 GAN 训练不稳定', '把 mode collapse 忽略掉', '只会说“生成图片”，不会讲 G 和 D 的互动关系', '不理解为什么要交替训练'],
        'mnemonic': 'G 造假，D 打假；你追我赶，越练越像。',
        'quiz': ['GAN 中 G 和 D 分别做什么？', '为什么 GAN 容易训练不稳定？', '什么是 mode collapse？'],
    },
    {
        'filename': '08_Diffusion_中文详细原理图.png',
        'title': 'Diffusion 扩散模型',
        'core': '先把真实数据逐步加噪声破坏，再训练模型学会一步步去噪，从纯噪声中反向生成新样本。',
        'flow': ['真实样本 x0', '前向过程逐步加噪 -> x1,x2...xT', '训练模型预测噪声', '推理时从随机噪声 xT 开始', '逐步去噪', '得到生成样本'],
        'layers': ['前向扩散：固定加噪过程', '去噪网络：通常是 U-Net/条件网络', '时间步编码：告诉模型当前噪声阶段', '反向采样：一步步恢复干净样本'],
        'mech': ['前向过程简单：不断加高斯噪声', '训练目标常是预测当前噪声 ε', '反向过程近似学习 p(x_{t-1}|x_t)', '从噪声出发可生成高质量样本'],
        'pitfalls': ['把前向加噪和反向生成顺序说反', '只会说“去噪”，不会说训练目标是预测噪声', '忽略时间步条件信息', '不知道为什么推理慢：因为要多步采样'],
        'mnemonic': '先加噪，再学去噪；从噪声回到图像。',
        'quiz': ['Diffusion 为什么能从噪声生成图片？', '训练时模型通常预测什么？', '为什么它比 GAN 更稳定但推理更慢？'],
    },
]


def wrap_text(text: str, width: int) -> list[str]:
    lines = []
    for para in text.split('\n'):
        if not para:
            lines.append('')
            continue
        cur = ''
        for ch in para:
            test = cur + ch
            if len(test) <= width:
                cur = test
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


def section_box(draw, x, y, w, title, body_lines, header_color, title_font, text_font):
    pad = 24
    header_h = 52
    temp_img = Image.new('RGB', (10, 10))
    temp_draw = ImageDraw.Draw(temp_img)
    text_h = 0
    for line in body_lines:
        wrapped = wrap_text(line, 34)
        for wl in wrapped:
            bbox = temp_draw.textbbox((0, 0), wl, font=text_font)
            text_h += (bbox[3] - bbox[1]) + 8
    h = header_h + text_h + pad * 2 + 10
    draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill='white', outline='#D7E3F4', width=3)
    draw.rounded_rectangle((x, y, x + w, y + header_h), radius=26, fill=header_color, outline=header_color)
    draw.rectangle((x, y + header_h - 26, x + w, y + header_h), fill=header_color)
    draw.text((x + 18, y + 11), title, font=title_font, fill='white')
    cy = y + header_h + pad - 4
    for line in body_lines:
        prefix = '' if title in ('核心一句话', '记忆口诀') else '• '
        wrapped = wrap_text(prefix + line, 34)
        for wl in wrapped:
            draw.text((x + 20, cy), wl, font=text_font, fill='#203040')
            bbox = draw.textbbox((x + 20, cy), wl, font=text_font)
            cy += (bbox[3] - bbox[1]) + 8
    return h


def make_card(info: dict):
    W, H = 1600, 2600
    img = Image.new('RGB', (W, H), '#F4F8FD')
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(BOLD_PATH, 52)
    sub_font = ImageFont.truetype(BOLD_PATH, 28)
    body_font = ImageFont.truetype(FONT_PATH, 26)
    small_font = ImageFont.truetype(FONT_PATH, 22)

    draw.rounded_rectangle((60, 40, W - 60, 180), radius=36, fill='#1F5AA6')
    draw.text((100, 70), info['title'], font=title_font, fill='white')
    draw.text((100, 130), '结构：核心一句话 → 整体流程 → 分层结构 → 关键机制 → 易错点 → 记忆口诀 → 自测问题', font=small_font, fill='#EAF3FF')

    h = section_box(draw, 70, 220, 1460, '核心一句话', [info['core']], '#2F80ED', sub_font, body_font)
    y = 220 + h + 24
    h1 = section_box(draw, 70, y, 1460, '整体流程', [' → '.join(info['flow'])], '#3BAE7A', sub_font, body_font)
    y += h1 + 24

    left_x, right_x, box_w = 70, 820, 710
    yl, yr = y, y
    hl = section_box(draw, left_x, yl, box_w, '分层结构', info['layers'], '#7B61FF', sub_font, body_font)
    hr = section_box(draw, right_x, yr, box_w, '关键机制', info['mech'], '#F2994A', sub_font, body_font)
    yl += hl + 24
    yr += hr + 24

    hl2 = section_box(draw, left_x, yl, box_w, '易错点', info['pitfalls'], '#EB5757', sub_font, body_font)
    hr2 = section_box(draw, right_x, yr, box_w, '记忆口诀', [info['mnemonic']], '#27AE60', sub_font, body_font)
    yl += hl2 + 24
    yr += hr2 + 24

    bottom_y = max(yl, yr)
    h3 = section_box(draw, 70, bottom_y, 1460, '自测问题', info['quiz'], '#9B51E0', sub_font, body_font)

    footer_y = bottom_y + h3 + 30
    draw.text((W - 520, footer_y), '用于快速理解 / 长期记忆 / 面试表达', font=small_font, fill='#6B7A90')
    final_h = min(H, footer_y + 70)
    img = img.crop((0, 0, W, final_h))
    out = OUT_DIR / info['filename']
    img.save(out)
    print(f'Saved {out}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for info in ALGORITHMS:
        make_card(info)
    print(f'全部中文详细原理图已输出到: {OUT_DIR}')


if __name__ == '__main__':
    main()
