没错，现在就把这一版 **“可直接按代码实现”的 Vision Encoder 结构**完整给你梳理出来。
 我会按照 **模块 → 尺度 → 具体卷积参数** 来写，你交给同学/CodeX 基本可以直接撸 PyTorch。

下面默认用 PyTorch 风格命名（Conv2d / BatchNorm2d / ReLU）。

------

## 0. 总体结构概览

### 输入输出形式（以 512×512 为例）

- 输入：`I`，形状 `(B, 3, H, W)`，例如 `(B, 3, 512, 512)`

- DefT 编码器输出 4 个多尺度特征：

  - `F2`: `(B, C2, H/4,  W/4 )` 例如 `(B, 64, 128, 128)`
  - `F3`: `(B, C3, H/8,  W/8 )` 例如 `(B,128,  64,  64)`
  - `F4`: `(B, C4, H/16, W/16)` 例如 `(B,256,  32,  32)`
  - `F5`: `(B, C5, H/32, W/32)` 例如 `(B,512,  16,  16)`

- DEM 各层增强后输出：

  - `F2'`: `(B, C, H/4,  W/4 )`
  - `F3'`: `(B, C, H/8,  W/8 )`
  - `F4'`: `(B, C, H/16, W/16)`
  - `F5'`: `(B, C, H/32, W/32)`

  其中 C 是统一通道数（推荐 256）。

- FPN 融合后输出单一特征图：

  - `F_def`: `(B, C, H/4, W/4)`

- 给 Adapter 的最终特征图：

  - `F_enc = Conv1×1(F_def)`: `(B, d, H/4, W/4)`，d 为 Adapter 的 `d_model`，比如 256 或 512。

------

## 1. 顶层模块：VisionEncoder

伪代码结构：

```python
class VisionEncoder(nn.Module):
    def __init__(self, deft_backbone, C=256, d_model=256):
        super().__init__()
        self.backbone = deft_backbone  # 已训练或自定义的 DefT 编码器

        # 各尺度通道对齐
        self.proj2 = ConvBNReLU(C2, C, k=1)
        self.proj3 = ConvBNReLU(C3, C, k=1)
        self.proj4 = ConvBNReLU(C4, C, k=1)
        self.proj5 = ConvBNReLU(C5, C, k=1)

        # 各尺度 DEM
        self.dem2 = DEM_SmallScale(C)   # 几何+纹理为主
        self.dem3 = DEM_MidScale(C)     # 几何+纹理为主
        self.dem4 = DEM_ColorScale(C)   # 颜色–湿度为主
        self.dem5 = DEM_ColorScale(C)   # 颜色–湿度为主

        # FPN 融合
        self.fpn4 = ConvBNReLU(2*C, C, k=3, s=1, p=1)
        self.fpn3 = ConvBNReLU(2*C, C, k=3, s=1, p=1)
        self.fpn2 = ConvBNReLU(2*C, C, k=3, s=1, p=1)

        # 输出到 Adapter 的投影
        self.out_proj = ConvBNReLU(C, d_model, k=1, s=1, p=0, act=False)

    def forward(self, x_rgb):
        # 1) DefT 编码器输出多尺度特征
        F2, F3, F4, F5 = self.backbone(x_rgb)

        # 2) 通道对齐
        P2 = self.proj2(F2)  # (B,C,H/4,W/4)
        P3 = self.proj3(F3)  # (B,C,H/8,W/8)
        P4 = self.proj4(F4)  # (B,C,H/16,W/16)
        P5 = self.proj5(F5)  # (B,C,H/32,W/32)

        # 3) 各尺度 DEM 增强
        F2p = self.dem2(P2)             # (B,C,H/4,W/4)
        F3p = self.dem3(P3)             # (B,C,H/8,W/8)
        F4p = self.dem4(P4, x_rgb)      # (B,C,H/16,W/16)
        F5p = self.dem5(P5, x_rgb)      # (B,C,H/32,W/32)

        # 4) FPN 自顶向下融合为 H/4 的 F_def
        U5 = F5p
        U4 = self.fpn4(torch.cat([F4p, F.interpolate(U5, scale_factor=2, mode="bilinear", align_corners=False)], dim=1))
        U3 = self.fpn3(torch.cat([F3p, F.interpolate(U4, scale_factor=2, mode="bilinear", align_corners=False)], dim=1))
        U2 = self.fpn2(torch.cat([F2p, F.interpolate(U3, scale_factor=2, mode="bilinear", align_corners=False)], dim=1))
        F_def = U2  # (B,C,H/4,W/4)

        # 5) 输出给 Adapter 的特征图
        F_enc = self.out_proj(F_def)  # (B,d_model,H/4,W/4)
        return F_enc, (F2p, F3p, F4p, F5p, F_def)
```

> 上面用的 `ConvBNReLU` 是一个小封装：Conv2d + BatchNorm2d + ReLU（可选）。

------

## 2. 小尺度 DEM2：细裂缝 + 细纹理增强

### 模块接口

```python
class DEM_SmallScale(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.geo = GeoBlockSmall(C)
        self.tex = TextureBlockSmall(C)
        self.fuse = ConvBNReLU(2*C, C, k=1, s=1, p=0)
    def forward(self, P2):  # P2: (B,C,H/4,W/4)
        G = self.geo(P2)   # (B,C,H/4,W/4)
        T = self.tex(P2)   # (B,C,H/4,W/4)
        F2p = self.fuse(torch.cat([G, T], dim=1))
        return F2p
```

### 几何增强块 GeoBlockSmall（重边缘、方向）

```python
class GeoBlockSmall(nn.Module):
    def __init__(self, C):
        super().__init__()
        # 多尺度 + 方向卷积
        self.conv1 = ConvBNReLU(C, C, k=3, s=1, p=1, d=1)  # 局部几何
        self.conv2 = ConvBNReLU(C, C, k=3, s=1, p=2, d=2)  # 空洞增强
        
        self.conv_h = ConvBNReLU(C, C, k=(1,3), s=1, p=(0,1))  # 水平敏感
        self.conv_v = ConvBNReLU(C, C, k=(3,1), s=1, p=(1,0))  # 垂直敏感

        self.fuse = ConvBNReLU(3*C, C, k=1, s=1, p=0)

    def forward(self, x):
        # x: (B,C,H,W)
        x1 = self.conv1(x)           # 局部
        x2 = self.conv2(x1)          # 多尺度
        
        xh = self.conv_h(x1)
        xv = self.conv_v(x1)
        xdir = xh + xv               # 方向结构
        
        cat = torch.cat([x2, xdir, x], dim=1)  # 原特征也拼进来
        out = self.fuse(cat)                  # (B,C,H,W)
        out = out + x                         # 残差
        return out
```

- 参数：
  - conv1, conv2: kernel=3, padding=1 / dilation=2；通道 C→C
  - conv_h, conv_v: 分别是 (1,3) 和 (3,1)，捕捉细长结构
  - fuse: 1×1 Conv，将 3C 压回 C

### 纹理增强块 TextureBlockSmall（细纹理 + 粗糙度）

```python
class TextureBlockSmall(nn.Module):
    def __init__(self, C):
        super().__init__()
        # 多核 depthwise 卷积
        self.dw3 = DWConvBNReLU(C, C, k=3, s=1, p=1)
        self.dw5 = DWConvBNReLU(C, C, k=5, s=1, p=2)
        self.fuse = ConvBNReLU(2*C, C, k=1, s=1, p=0)

        # 粗糙度估计
        self.rough_conv = nn.Conv2d(C, C, kernel_size=1)

    def forward(self, x):
        # x: (B,C,H,W)
        t3 = self.dw3(x)
        t5 = self.dw5(x)
        t0 = self.fuse(torch.cat([t3, t5], dim=1))  # 纹理基底

        # 局部均值 + 残差
        mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        r  = torch.abs(x - mu)
        rough = torch.sigmoid(self.rough_conv(r))   # (B,C,H,W), 0~1

        out = t0 * (1.0 + 0.5 * rough)              # 0.5 可初始设定，可学也行
        return out
```

- `DWConvBNReLU`：depthwise Conv2d(groups=C) + BN + ReLU
- 这种结构同时捕捉不同感受野的纹理，并用 `rough` 强调局部对比度大的区域。

------

## 3. 中尺度 DEM3：中等裂缝 + 剥落粗糙

接口基本一致，只是感受野稍大。

```python
class DEM_MidScale(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.geo = GeoBlockMid(C)
        self.tex = TextureBlockMid(C)
        self.fuse = ConvBNReLU(2*C, C, k=1, s=1, p=0)

    def forward(self, P3):  # (B,C,H/8,W/8)
        G = self.geo(P3)
        T = self.tex(P3)
        F3p = self.fuse(torch.cat([G, T], dim=1))
        return F3p
```

### 几何块 GeoBlockMid：增加更大 dilation

```python
class GeoBlockMid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = ConvBNReLU(C, C, k=3, s=1, p=1, d=1)
        self.conv2 = ConvBNReLU(C, C, k=3, s=1, p=2, d=2)
        self.conv3 = ConvBNReLU(C, C, k=3, s=1, p=3, d=3)
        self.fuse = ConvBNReLU(4*C, C, k=1, s=1, p=0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        cat = torch.cat([x1, x2, x3, x], dim=1)
        out = self.fuse(cat)
        return out + x
```

### 纹理块 TextureBlockMid：更大窗口粗糙度

```python
class TextureBlockMid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw3 = DWConvBNReLU(C, C, k=3, s=1, p=1)
        self.dw5 = DWConvBNReLU(C, C, k=5, s=1, p=2)
        self.fuse = ConvBNReLU(2*C, C, k=1, s=1, p=0)
        self.rough_conv = nn.Conv2d(C, C, kernel_size=1)
    def forward(self, x):
        t3 = self.dw3(x)
        t5 = self.dw5(x)
        t0 = self.fuse(torch.cat([t3, t5], dim=1))

        mu = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2) # 窗口更大
        r  = torch.abs(x - mu)
        rough = torch.sigmoid(self.rough_conv(r))

        out = t0 * (1.0 + 0.5 * rough)
        return out
```

------

## 4. 大尺度 DEM4 / DEM5：颜色–湿度增强

这两个尺度结构相同，只是空间尺寸不同；为了重用代码，可以做一个 `DEM_ColorScale` 模块。

### 接口设计

```python
class DEM_ColorScale(nn.Module):
    def __init__(self, C, down_ratio):
        super().__init__()
        # down_ratio: 从原图下采样到当前尺度的倍数，例如 16 或 32
        self.down_ratio = down_ratio

        # Lab 颜色嵌入: 3 -> C
        self.color_embed = ConvBNReLU(3, C, k=3, s=down_ratio, p=1)

        # 融合 + SE
        self.fuse = ConvBNReLU(2*C, C, k=3, s=1, p=1)
        self.se_fc1 = nn.Linear(C, C//16)
        self.se_fc2 = nn.Linear(C//16, C)

        # LF/HF 分支
        self.lf_conv = ConvBNReLU(C, C, k=1, s=1, p=0)
        self.hf_conv1 = ConvBNReLU(C, C, k=3, s=1, p=1)
        self.hf_conv2 = nn.Conv2d(C, C, kernel_size=1)
        self.att_conv = nn.Conv2d(2*C, C, kernel_size=1)

    def forward(self, Ps, x_rgb):
        # Ps: (B,C,Hs,Ws), x_rgb: (B,3,H,W)
        # 1) Lab 转换由外部数据预处理完成：x_lab: (B,3,H,W)
        # 若在网络内做，可以在 Dataset 或 forward 前转换

        x_lab = rgb_to_lab_like(x_rgb)  # 伪函数，实际可在 Dataset 中预处理

        # 2) 颜色嵌入 & 下采样
        Ls = self.color_embed(x_lab)    # (B,C,Hs,Ws)，stride=down_ratio

        # 3) 与特征融合 + SE
        C_in = torch.cat([Ps, Ls], dim=1)
        C0 = self.fuse(C_in)           # (B,C,Hs,Ws)

        # SE 通道注意力
        B, C, Hs, Ws = C0.shape
        z = F.adaptive_avg_pool2d(C0, 1).view(B, C)   # (B,C)
        e = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(z))))  # (B,C)
        e = e.view(B, C, 1, 1)
        C1 = C0 * e

        # 4) LF/HF 分支
        # LF: 大核池化可以用 AvgPool 然后上采样
        k = 7
        pad = k // 2
        LF_pool = F.avg_pool2d(C1, kernel_size=k, stride=1, padding=pad)
        LF = self.lf_conv(LF_pool)     # 保持 Hs,Ws 不变

        # HF: 局部变化
        HF = self.hf_conv2(self.hf_conv1(C1))

        # 注意力 mask
        att_in = torch.cat([LF, HF], dim=1)   # (B,2C,Hs,Ws)
        A = torch.sigmoid(self.att_conv(att_in))  # (B,C,Hs,Ws)

        out = C1 * (1.0 + 0.5 * A)
        return out
```

在 VisionEncoder 中这样实例化：

```python
self.dem4 = DEM_ColorScale(C, down_ratio=16)
self.dem5 = DEM_ColorScale(C, down_ratio=32)
...
F4p = self.dem4(P4, x_rgb)  # H/16
F5p = self.dem5(P5, x_rgb)  # H/32
```

> `rgb_to_lab_like` 你可以在数据预处理阶段实现，也可以在网络前半部分实现一次，作为额外输入；上面是逻辑示意。

------

## 5. FPN 融合与 Adapter 投影

### FPN 融合部分（已经在 VisionEncoder 里给了）

这里强调一下形状流动（以 512×512 为例）：

- `F5p`: (B,C,16,16)
- `up(F5p)`: (B,C,32,32) 和 `F4p` concat → (B,2C,32,32) → `U4`:(B,C,32,32)
- `up(U4)`: (B,C,64,64) 和 `F3p` concat → (B,2C,64,64) → `U3`:(B,C,64,64)
- `up(U3)`: (B,C,128,128) 和 `F2p` concat → (B,2C,128,128) → `U2`:(B,C,128,128) = `F_def`

### Adapter 投影层

```python
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s,
                              padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

Adapter 接口：

- `F_enc = self.out_proj(F_def)`
- 形状 `(B, d_model, H/4, W/4)`
- 后续在多模态部分直接 flatten：

```python
B, C, H4, W4 = F_enc.shape
vis_tokens = F_enc.view(B, C, H4*W4).permute(0, 2, 1)  # (B, N, d_model)
```

------

## 6. 各类特征提取需求与结构对应（简要对照）

- **裂缝（细长线）**
  - DEM2/DEM3 的 `GeoBlockSmall/Mid`：多尺度 3×3+dilated、方向卷积，对高梯度、线状结构敏感。
- **剥落 + 粗糙纹理 + 骨料暴露**
  - DEM2/DEM3 的 `TextureBlock`：多核 depthwise 卷积提纹理，高/低频纹理 + 粗糙度 (|F-μ|) 权重，突出粗糙区域。
- **锈蚀 / 渗漏 / 析出（白华）**
  - DEM4/DEM5：从原图 Lab 空间抽取颜色/亮度信息，和 DefT 深层特征融合，通过 SE 挑选颜色通道；
  - 再用 LF（大核平均） + HF（3×3 conv）生成注意力，强调大块变色 + 边界渐变区域。
- **多尺度统一与对 Adapter 友好**
  - F2'～F5' 保留各自侧重增强；FPN 将它们统一融合成 `F_def (H/4)`，既有细节又有全局上下文；
  - 对 Adapter 只暴露一张标准特征图，接口简单，方便后续模态对齐。

------

如果你愿意，下一步我可以帮你把：

- **这一整套设计写成正式英文/中文论文的「3.x 视觉编码器结构」小节**，
- 或者在这个结构基础上，再补一个**专门给 Decoder 的多尺度输出设计**（如果你打算用同一个 Encoder 做分割实验）。