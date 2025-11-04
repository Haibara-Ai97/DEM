## 二、整体结构概览（论文可以画成一张图）

在 3.4 Adapter 小节里，可以这样整体描述：

> 视觉编码器 + DEM 输出多尺度融合特征图 (F \in \mathbb{R}^{B\times H \times W \times C}) 后，本节设计一个基于 SEA 的语义对齐 Adapter。该模块包含两条路径：
>  1）**主路径**：将 (F) 转换为 LLM 输入空间中的视觉 token 序列；
>  2）**对齐监督路径**：利用在混凝土缺陷数据上微调过的 CLIP 提取 patch 级语义标签，对 Adapter 输出进行 SEA 式对比学习监督，实现 token 级模态对齐。

流程（文字版）：

1. 前文 Encoder+DEM 输出最终特征 (F)。
2. **Adapter 主体**：
   - 通过 (1\times1) 卷积把通道数转到 (d_{\text{llm}})；
   - 展平为 (m = H\cdot W) 个 token，输入到 per-token MLP Adapter；
   - 得到视觉 token 序列 (X_v \in \mathbb{R}^{B\times m \times d_{\text{llm}}})。
3. **域自适应 CLIP 分支**：
   - 使用在混凝土缺陷数据集上微调后的 CLIP ((f_c, h_c)) 对同一张图像提取 patch 级语义标签（多候选词 + 相似度）；
   - 将采样得到的标签 (w_{i}) 映射到 LLM 的词嵌入空间，得到标签向量 (t_{i})。
4. **SEA 对齐损失**：
   - 对应位置的 Adapter 输出 token (x^v_i) 与标签向量 (t_i) 构成正样本；
   - 同一 batch 内其他标签向量 / 视觉 token 构成负样本；
   - 通过对称 InfoNCE 对比损失 (\mathcal{L}_a) 约束 Adapter，把视觉 token 拉近到正确的语义词向量附近。

------

## 三、模型结构细化设计

### 3.4.1 输入特征与统一网格

记前文 3.3 节 Encoder+DEM 输出的最终融合特征为：

[
 F \in \mathbb{R}^{B\times H\times W\times C}
 ]

其中 (B) 为 batch 大小，(H,W) 为统一的空间分辨率，(C) 为通道数。为了和 CLIP patch 网格对齐，我们做：

1. **空间对齐**
   - 选定 CLIP 视觉编码器的 patch 网格大小为 ((H_c, W_c))；
   - 将 (F) 通过双线性插值 resize 到 ((H_c, W_c))：
      [
      F' = \text{Resize}(F) \in \mathbb{R}^{B\times H_c\times W_c\times C}
      ]
2. 将 (F') 展平为 patch 序列：
    [
    V_0 = \text{Flatten}(F') \in \mathbb{R}^{B\times m \times C},\quad m = H_c\cdot W_c
    ]

后续 SEA 对齐和 CLIP patch 标签都在这个统一的 patch 网格上进行一一监督。

------

### 3.4.2 Adapter 主体：per-token MLP 投影

目标是把每个视觉 patch 特征投到 LLM 词向量维度 (d_{\text{llm}}) 上，并保留一定的非线性表达能力。结构设计如下：

1. **通道投影（Conv1×1 等价于线性层）**

[
 V = \text{Conv}*{1\times1}(F') \in \mathbb{R}^{B\times H_c\times W_c\times d*{\text{llm}}}
 ]
 [
 V = \text{Flatten}(V) \in \mathbb{R}^{B\times m \times d_{\text{llm}}}
 ]

Conv 的权重可记为 (W_{\text{proj}} \in \mathbb{R}^{C \times d_{\text{llm}}})。

1. **Adapter MLP 结构**

对第 (i) 个 patch 的特征向量 (v_i \in \mathbb{R}^{d_{\text{llm}}})，定义：

- 预归一化：
   [
   \hat{v}_i = \text{LayerNorm}(v_i)
   ]
- 两层全连接 + GELU 激活：
   [
   h_i = \text{GELU}(W_1 \hat{v}*i + b_1), \quad W_1\in\mathbb{R}^{d*{\text{mid}}\times d_{\text{llm}}}
   ]
   [
   \Delta_i = W_2 h_i + b_2,\quad W_2\in\mathbb{R}^{d_{\text{llm}}\times d_{\text{mid}}}
   ]
- 残差连接（保持稳定性）：
   [
   x^v_i = v_i + \Delta_i \in \mathbb{R}^{d_{\text{llm}}}
   ]

于是，Adapter (g_\theta) 的整体形式是：

[
 X_v = g_\theta(F) \in \mathbb{R}^{B\times m\times d_{\text{llm}}}
 ]

这和 SEA / LLaVA 中“MLP Adapter 将视觉 patch 映射到 LLM 输入空间”的定义是一致的，只是我们加了 LayerNorm 和残差以增强训练稳定性。

> **实现建议**：
>
> - 取 (d_{\text{mid}} = 2d_{\text{llm}})（比如 LLM 维度 4096 则 d_mid 约 8192）；
> - 在代码里 Adapter 可以写成一个 `nn.Sequential(LayerNorm, Linear, GELU, Linear)` + 残差。

------

### 3.4.3 域自适应 CLIP 分支

这一部分在训练时存在，推理时不参与，仅作为 Adapter 的“教师”。

1. **域微调 CLIP**

- 基础模型：预训练 CLIP（例如 ViT-B/16 或 ViT-L/14 结构）。
- 训练数据：混凝土缺陷图像–文本对 (\mathcal{D}_{\text{concrete}} = {(I_k, T_k)})，文本包含人工标注描述以及由大模型生成的缺陷描述。
- 损失：标准 CLIP 对比损失（image–text 双向 InfoNCE）。

得到域自适应的视觉编码器 (f_c) 和文本编码器 (h_c)。

1. **构建领域词表**

- 以你的业务为中心，构建 defect 词表 (W_{\text{def}} = {w_1, \dots, w_q})：
  - 缺陷类型：crack, hairline crack, spalling, honeycomb, efflorescence, rust stain, water leakage, rebar exposure…
  - 定性属性：fine, wide, dense, scattered, severe, slight…
  - 颜色/材质：dark streak, whitish, damp, mold, steel bar…
- 可在预训练语料的基础上筛选出与混凝土/结构工程相关的高频词，类似 SEA 从预训练语料中构造约 4M 规模词表的做法。

1. **CLIP patch 特征与文本特征**

对一张图像 (I)：

[
 Z = f_c(I) \in \mathbb{R}^{m \times d_c} \quad (\text{CLIP patch 特征})
 ]
 [
 T_c = h_c(W_{\text{def}}) \in \mathbb{R}^{q \times d_c} \quad (\text{领域词表的文本特征})
 ]

------

### 3.4.4 语义标签提取与采样（SEA 风格）

参考 SEA 的 Eq.(6)-(9)，对每个 patch 特征 (z_i) 与所有词向量 (t^c_j) 计算余弦相似度，选出 top-(n) 个语义标签。

1. **候选标签与相似度**

[
 \text{sim}_{ij} = \cos(z_i, t^c_j)
 ]

对每个 patch (i)，取相似度最高且大于 0 的前 (n) 个词，构成：

[
 L_i = [w_{i1},\dots,w_{in}],\quad S_i = [s_{i1},\dots,s_{in}]
 ]

1. **相似度归一化 + 加权采样**

和 SEA 一样，为保留连续语义但又为每个 patch 选出一个主标签：

[
 S^\text{norm}*i = \frac{S_i}{\sum*{k=1}^n s_{ik}}
 ]

从 (L_i) 中按照 (S^\text{norm}_i) 分布采样一个标签 (w_i) 作为该 patch 的“代表语义”。

1. **局部窗口采样（Localized Sampling）**

为了避免同一图像中大量极相似 patch 参与对比学习导致训练不稳定，沿用 SEA 的局部窗口策略：

- 将 patch 网格划分为 (k\times k) 小窗口（例如 (k=2)）；
- 每个窗口中仅随机保留 1 个 patch 做对比学习；
- 同一 batch 内若多个 patch 采样到相同标签，仅保留 1 个。

最终得到一批带标签的视觉 token：

[
 {(x^v_1, w_1), \dots, (x^v_N, w_N)}
 ]

这里 (x^v_i) 是 **Adapter 输出**，而不是 CLIP 的特征。

1. **标签在 LLM 词向量空间中的表示**

对每个标签 (w_i)，可能由多 token 构成（如 “wide crack”）。
 用 LLM 的词嵌入层 (\Psi) 编码并取平均：

[
 t_i = \frac{1}{M}\sum_{k=1}^M \Psi(w_i^{(k)}) \in \mathbb{R}^{d_{\text{llm}}}
 ]

------

### 3.4.5 SEA 对齐损失与总损失

对齐损失直接沿用 SEA 的对称 InfoNCE 形式：

设 batch 内共有 (N) 个参与对比的 patch–标签对 ((x^v_i, t_i))，定义归一化余弦相似度：

[
 \phi(a,b) = \frac{a}{|a|_2}\cdot \frac{b}{|b|_2}
 ]

则对齐损失为：

[
 \mathcal{L}*a = -\frac{1}{2N}\sum*{i=1}^N
 \left[
 \log \frac{\exp(\phi(x^v_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\phi(x^v_i, t_j)/\tau)}
 +
 \log \frac{\exp(\phi(t_i, x^v_i)/\tau)}{\sum_{j=1}^N \exp(\phi(t_i, x^v_j)/\tau)}
 \right]
 ]

其中 (\tau) 为可学习温度参数。

> **直观解释写进论文时可以一句话带过**：
>  Adapter 输出的每个视觉 token 会被“拉近”到它对应的缺陷语义标签在 LLM 词向量空间中的位置，同时远离其他无关标签，从而在 LLM 输入空间内完成 token 级模态对齐。

------

### 3.4.6 训练流程小结（方便写成一小段文字）

你可以在 3.4 的最后给一个训练流程概述：

1. **阶段 0：CLIP 域自适应预训练**
   - 在混凝土缺陷数据集上微调 CLIP，获得领域视觉–文本编码器 ((f_c, h_c))。
2. **阶段 1：SEA-Adapter 预训练（本节重点）**
   - 冻结 LLM 和 CLIP，仅训练视觉 Encoder+DEM 的高层 & Adapter；
   - 对每张图像：
     - 用自定义 Encoder+DEM 得到特征 (F)，经 Adapter 得到视觉 token (X_v)；
     - 用域自适应 CLIP + 领域词表 (W_{\text{def}}) 提取 patch 语义标签 (w_i)；
     - 用 LLM 词嵌入层 (\Psi) 得到标签向量 (t_i)；
     - 计算 (\mathcal{L}_a)（可结合 (\mathcal{L}_g)）更新 Adapter 和高层视觉编码器参数。
3. **阶段 2：整 MLLM 指令微调（在第二部分详细展开）**
   - 以已经对齐好的 Adapter + 视觉编码器为初始化，进行图文问答/结构评估任务的指令微调。

------

这样写完之后，你的“Adapter 小节”在论文里就会非常完整：

- **结构上**：给出了具体的 Conv1×1 + per-token MLP + Residual 的网络结构和维度设计；
- **机制上**：说明了如何用“域自适应 CLIP + 领域词表”在混凝土缺陷场景下实现 SEA 式 token-level 监督；
- **训练上**：给出了损失函数公式和多阶段训练流程。