# SpectrumReconstruction 架构文档

**Chenyu Huang**$^{*\dagger}$

$^*$*State Key Laboratory of Infrared Physics, Shanghai Institute of Technical Physics, Chinese Academy of Sciences*

$^\dagger$*University of Chinese Academy of Sciences*

[TOC]

---

## 一、设计目标

`SpectrumReconstruction` 旨在对基于**可调谐半导体光电探测器**的计算光谱重构方案进行完整的物理仿真。其核心思路是：

> 通过改变探测器的工作状态（偏置电压、带隙能量等），使同一个探测器在不同状态下具有不同的光谱响应度，从而将多次测量的探测器输出信号解算为入射光谱。

模块在设计上追求以下目标：

- **物理准确**：响应度、黑体辐射等均基于严格的物理公式；
- **高性能**：核心计算路径全程使用 NumPy 广播+矩阵运算，关键物理函数通过 Numba `@njit` 加速；
- **分层解耦**：物理建模、数据流转、重构求解三个关注点分属不同层次，互不耦合；
- **易扩展**：基函数、偏置模式、正则化方法均可通过参数切换，无需修改核心代码。

---

## 二、物理模型

### 2.1 前向模型

设探测器在偏置状态 $b$ 下的光谱响应度为 $R(\lambda, b)$，入射光谱为 $S(\lambda)$，则探测器的输出信号为：

$$y(b) = \int R(\lambda, b) \cdot S(\lambda)\, d\lambda$$

离散化后写成矩阵形式：

$$\mathbf{y} = \mathbf{R}^{\top} \mathbf{s}$$

其中 $\mathbf{R} \in \mathbb{R}^{N_\lambda \times N_b}$ 为响应度矩阵，$\mathbf{s} \in \mathbb{R}^{N_\lambda}$ 为入射光谱向量，$\mathbf{y} \in \mathbb{R}^{N_b}$ 为探测器输出向量。

### 2.2 光谱表示与训练矩阵

假设入射光谱可以由一组基函数 $\{f_j(\lambda)\}$ 的线性组合来表示：

$$S(\lambda) = \sum_j a_j \cdot f_j(\lambda)$$

基函数目前支持两类：
- **高斯基**：$f_j(\lambda) = G(\lambda, \mu_j, \sigma)$，训练集覆盖一系列中心波长 $\{\mu_j\}$；
- **黑体基**：$f_j(\lambda) = B(\lambda, T_j)$，训练集覆盖一系列温度 $\{T_j\}$。

将所有基函数的探测器响应预计算为训练矩阵：

$$\mathbf{P}_{ij} = \mathbf{R}^{\top}_{i \cdot} \cdot \mathbf{f}_{j}
\quad \Leftrightarrow \quad
\mathbf{P} = \mathbf{R}^{\top} \mathbf{F}$$

其中 $\mathbf{F} \in \mathbb{R}^{N_\lambda \times N_j}$ 为基函数矩阵，$\mathbf{P} \in \mathbb{R}^{N_b \times N_j}$ 为训练响应矩阵（代码中称为 `response_pivot`）。

### 2.3 逆问题与正则化求解

给定未知光谱的探测器输出 $\mathbf{y}_{\text{test}} \in \mathbb{R}^{N_b}$，求解系数向量 $\mathbf{a}$ 使得：

$$\mathbf{P} \cdot \mathbf{a} \approx \mathbf{y}_{\text{test}}$$

由于该系统通常是欠定或病态的，模块提供多种带正则化的求解器：

| 方法名 | 目标函数 |
|---|---|
| `normal` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2$ |
| `l1` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{a}\|_1$ |
| `l2` / `Ridge` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{a}\|_2^2$ |
| `ElasticNet` | $\min \|\mathbf{P}\mathbf{a} - \mathbf{y}\|_2^2 + \lambda[\alpha\|\mathbf{a}\|_1 + (1-\alpha)\|\mathbf{a}\|_2^2]$ |

求解后光谱重构结果为 $\hat{S}(\lambda) = \sum_j a_j f_j(\lambda)$。

---

## 三、模块层次结构

模块从底层到顶层分为三个层次：

```
┌─────────────────────────────────────────────────────────┐
│              Layer 3: 端到端模拟层                        │
│         SpectrumReconstructionSimulation                 │
│  （参数配置 → 建模 → 训练 → 求解 → 可视化 一体化封装）      │
└──────────────────────────┬──────────────────────────────┘
                           │ 调用
┌──────────────────────────▼──────────────────────────────┐
│              Layer 2: 物理建模与数据流层                   │
│   SpectrumReconstructionAdvance  +  SpectrumReconstructionBasic  │
│  （探测器建模 / 光谱建模 / 响应矩阵 / 线性回归求解）          │
└──────────────────────────┬──────────────────────────────┘
                           │ 调用
┌──────────────────────────▼──────────────────────────────┐
│              Layer 1: 物理函数库层                         │
│                      Utility                             │
│  （blackbody / gaussian / smooth_responsivity / ...）    │
└─────────────────────────────────────────────────────────┘
```

### 3.1 Layer 1 — Utility（物理函数库）

文件：`src/SpectrumReconstruction/Utility.py`

提供无状态的物理计算原语，所有函数均支持 NumPy 数组广播：

| 函数 | 功能 |
|---|---|
| `blackbody(λ, T)` | 普朗克黑体辐射，对数空间计算，`@njit` |
| `gaussian(λ, μ, σ)` | 非归一化高斯函数，`@njit` |
| `ideal_responsivity(λ, Eg, η)` | 理想阶跃响应度 |
| `smooth_responsivity(λ, Eg, Δλ, η)` | Sigmoid 平滑响应度，`@njit` |
| `gaussian_spectrum_sum(λ, μ, σ, α)` | 多分量高斯叠加光谱 |
| `blackbody_spectrum_sum(λ, T, α)` | 多分量黑体叠加光谱 |
| `fast_matmul(A, B)` | 矩阵乘法 $A^{\top}B$ |

> **设计说明**：高频调用的物理核函数使用 Numba `@njit` 编译，首次调用有 JIT 编译延迟，后续调用接近原生速度。所有函数均支持标量与矩阵的混合广播，以支持上层的批量向量化运算。

### 3.2 Layer 2 — 物理建模与数据流层

#### 3.2.1 `SpectrumReconstructionAdvance`

文件：`src/SpectrumReconstruction/SpectrumReconstructionAdvance.py`

负责探测器与入射光谱的物理建模，以及响应矩阵的计算。

**类关系：**

```
IdealSemiconductorPhotoDetector          IncidentSpectrum
        │                                       │
        │ responsivity                          │ spectrum
        │ (N_λ × N_b)                           │ (N_λ × N_j)
        └──────────────┬────────────────────────┘
                       │
               simulate_response_matrix
                       │
               response_pivot (N_b × N_j)
                       │
               SpectrumReconstructionBasicHighPerformance
```

**`IdealSemiconductorPhotoDetector`**

核心是 `responsivity` 属性（`cached_property`），利用广播一次性计算所有偏置下的响应度矩阵，避免 for 循环：

```python
# normal_move 模式下的向量化计算
lambda_matrix = wavelength[:, None] - bias_array[None, :]  # (N_λ, N_b)
e_g_matrix    = full((N_λ, N_b), self.e_g)
responsivity  = smooth_responsivity(lambda_matrix, e_g_matrix, delta_lambda, eta)
```

支持三种偏置模式，各模式改变的物理量不同：

| 模式 | 改变量 | 效果 |
|---|---|---|
| `normal_move` | 有效波长 $\lambda_{\text{eff}} = \lambda - b$ | 响应度曲线整体平移 |
| `increase_band_gap` | 带隙 $E_g' = hc/(\lambda_g + b)$ | 截止波长向右移动 |
| `decrease_eta` | 带隙同上，同时 $\eta' = 1/(1+b/\lambda_g)$ | 截止波长移动且峰值降低 |

**`IncidentSpectrum`**

同样使用广播批量生成所有参数对应的光谱矩阵：

```python
# 高斯模式
spectrum_matrix = gaussian(wavelength[:, None], mu[None, :], sigma)  # (N_λ, N_μ)
# 黑体模式
spectrum_matrix = blackbody(wavelength[:, None], T[None, :])          # (N_λ, N_T)
```

**`SimulationSpectrum`**

接受任意可调用的光谱函数，通过 `set_spectrum(**kwargs)` 延迟生成未知光谱，与 `IncidentSpectrum` 解耦，专用于表示待重构的测试光谱。

#### 3.2.2 `SpectrumReconstructionBasic`

文件：`src/SpectrumReconstruction/SpectrumReconstructionBasic.py`

提供两个实现：

| 类 | 输入格式 | 适用场景 |
|---|---|---|
| `SpectrumReconstructionBasic` | `pandas.DataFrame`（长格式） | 实验数据，支持缺失值自动清理 |
| `SpectrumReconstructionBasicHighPerformance` | `numpy.ndarray`（已透视矩阵） | 仿真数据，零 DataFrame 开销 |

`SpectrumReconstructionSimulation` 内部使用高性能版本。

**`_clean_pivot_training_data`** 实现了一种贪心缺失值清理算法：每轮删除完整度最低的行或列，直到矩阵无缺失值，用于处理实验采集中因设备故障导致的不完整训练数据。

**`_linear_regression`** 统一封装了 12 种求解方法，通过 `match` 语句分发，支持 NumPy OLS、SciPy L1 优化以及 scikit-learn 全系列正则化回归器。

### 3.3 Layer 3 — `SpectrumReconstructionSimulation`（端到端模拟层）

文件：`src/SpectrumReconstruction/SpectrumReconstructionSimulation.py`

将 Layer 2 的所有组件串联为单一接口，内部构造顺序如下：

```
__init__()
    │
    ├─ 1. 构造 IdealSemiconductorPhotoDetector
    │      → 计算并缓存响应度矩阵 R (N_λ × N_b)
    │
    ├─ 2. 构造 IncidentSpectrum
    │      → 计算基函数矩阵 F (N_λ × N_j)
    │
    ├─ 3. simulate_response_matrix(R, F)
    │      → 训练响应矩阵 P = R^T F (N_b × N_j)
    │         存储为 self.response_pivot
    │
    └─ 4. 构造 SpectrumReconstructionBasicHighPerformance
           → 以 P 作为训练数据初始化求解器

reconstruct_spectrum(simulation_spectrum, method)
    │
    ├─ simulate_unknown_response(detector, simulation_spectrum)
    │      → 未知光谱响应向量 y_test (N_b,)
    │
    └─ spectrum_reconstruction.reconstruct_spectrum(y_test, method)
           → 求解系数 a，存储于 spectrum_reconstruction.a
```

---

## 四、数据流全景

```
参数配置
(bias_array, wavelength_array, e_g_ev, ...)
        │
        ▼
┌───────────────────────┐    ┌──────────────────────┐
│ IdealSemiconductor    │    │   IncidentSpectrum    │
│ PhotoDetector         │    │  (gaussian/blackbody) │
│                       │    │                       │
│ R: (N_λ × N_b)        │    │  F: (N_λ × N_j)       │
└──────────┬────────────┘    └──────────┬────────────┘
           │                            │
           └────────────┬───────────────┘
                        │ fast_matmul: R^T @ F
                        ▼
              ┌──────────────────┐
              │  response_pivot  │
              │  P: (N_b × N_j)  │  ← 训练矩阵
              └────────┬─────────┘
                       │
        ┌──────────────┴──────────────────┐
        │                                 │
        ▼                                 ▼
┌──────────────────┐          ┌───────────────────────┐
│ SimulationSpectrum│          │   线性回归求解器        │
│ s: (N_λ,)        │          │  P · a ≈ y_test        │
└────────┬─────────┘          │  方法：normal/l1/l2/   │
         │ fast_matmul         │       ElasticNet/...  │
         │ R^T @ s             └───────────┬───────────┘
         ▼                                 │
 y_test: (N_b,) ────────────────────────► a: (N_j,)
                                           │
                                           ▼
                               S(λ) = Σ aⱼ·fⱼ(λ)
                               重构光谱
```

---

## 五、关键设计决策

### 5.1 全程向量化，避免 Python 循环

响应度矩阵和光谱矩阵的计算均利用 NumPy 广播一次完成，矩阵乘法使用 `A.T @ B`。在典型配置（1501 个偏置点 × 28001 个波长点 × 2801 个高斯基）下，向量化相比逐列迭代快约 100 倍。

### 5.2 `cached_property` 缓存昂贵计算

`IdealSemiconductorPhotoDetector.responsivity` 被标注为 `cached_property`。响应度矩阵仅在第一次访问时计算，后续直接读取缓存。这在多次调用 `reconstruct_spectrum` 时（例如对比不同正则化参数）可避免重复计算。

### 5.3 两个重构类，按场景选择

| | `SpectrumReconstructionBasic` | `SpectrumReconstructionBasicHighPerformance` |
|---|---|---|
| 训练数据格式 | 长格式 DataFrame | NumPy ndarray（已透视） |
| 缺失值处理 | 支持（贪心清理算法） | 不支持 |
| 内存/速度 | 较高开销 | 低开销 |
| 典型用途 | 真实实验数据 | 仿真流水线 |

### 5.4 基函数与偏置模式的正交设计

基函数（gaussian/blackbody）和偏置模式（normal_move/increase_band_gap/decrease_eta）是两个独立维度，可自由组合，共 6 种配置，均通过同一套接口驱动，无需修改核心代码。

### 5.5 可见盲模拟

通过模块级全局参数 `visible_blind_cutoff_parameter`（由 `change_visible_blind_cutoff_parameter()` 设置）控制短波区域的响应度乘数，配合构造函数的 `visible_blind_cutoff` 阈值，可模拟探测器的可见盲特性而无需重新计算整个响应度矩阵。

---

## 六、文件结构

```
src/SpectrumReconstruction/
├── __init__.py                         # 公共接口导出
├── Utility.py                          # Layer 1：物理函数库
├── SpectrumReconstructionAdvance.py    # Layer 2：探测器与光谱建模
├── SpectrumReconstructionBasic.py      # Layer 2：线性回归求解器
└── SpectrumReconstructionSimulation.py # Layer 3：端到端模拟封装

tests/
├── conftest.py                         # 共享 fixtures
├── test_utility.py                     # Utility 函数单元测试
├── test_advance.py                     # 探测器与光谱建模测试
├── test_reconstruction_basic.py        # 重构求解器测试
└── test_simulation.py                  # 端到端集成测试

docs/
├── architecture_CN.md                  # 本文档
├── Module_and_API_Reference_CN.md      # 中文 API 参考
└── Module_and_API_Reference.md         # 英文 API 参考

examples/
├── example.ipynb                       # SpectrumReconstructionBasic 使用示例
├── data_training_example.csv           # 示例训练数据
└── data_testing_example.csv            # 示例测试数据
```

---

## 七、扩展指南

### 新增偏置模式

在 `IdealSemiconductorPhotoDetector` 的 `responsivity` 属性和 `_responsivity_func` 方法中，各添加一个 `case` 分支，计算对应的 `lambda_matrix`、`e_g_matrix` 和 `eta`（局部变量，不得修改 `self.eta`）。

### 新增正则化方法

在 `SpectrumReconstructionBasic._linear_regression` 函数的 `match method` 语句中添加新的 `case` 分支，返回 `numpy.ndarray` 系数即可。

### 新增基函数

1. 在 `Utility.py` 中实现新函数，支持 NumPy 数组广播；
2. 在 `IncidentSpectrum.__init__` 的 `match` 语句中添加新模式；
3. 在 `SpectrumReconstructionSimulation.__init__` 的对应位置添加初始化逻辑。
