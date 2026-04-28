<!-- SpectrumReconstruction -->

<!--suppress HtmlDeprecatedAttribute -->
<div align="center" style="text-align:center">
   <h1> SpectrumReconstruction </h1>
   <p>
      光谱重构与光电探测器模拟 Python 模块<br>
      <code><b> v0.2 </b></code>
   </p>
   <p>
      <img alt="GitHub Top Language" src="https://img.shields.io/github/languages/top/hcy2206/SpectrumReconstruction?label=Python">
      <img alt="GitHub License" src="https://img.shields.io/github/license/hcy2206/SpectrumReconstruction?label=License"/>
   </p>
   <p>
      <a href="readme.md">English</a> | 中文
   </p>
</div>

## 简介

**SpectrumReconstruction** 是一个用于模拟基于半导体光电探测器响应的计算光谱重构全流程的 Python 模块。

主要功能：
- **光电探测器建模** &mdash; 模拟理想半导体光电探测器的光谱响应度，支持可调带隙、量子效率、多种偏置模式（`normal_move`、`increase_band_gap`、`decrease_eta`）及可见盲特性。
- **入射光谱生成** &mdash; 使用高斯函数或黑体辐射模型生成训练光谱。
- **响应矩阵计算** &mdash; 计算探测器对一组已知入射光谱的响应矩阵。
- **光谱重构** &mdash; 通过线性回归及可选的正则化方法（OLS、Lasso、Ridge、ElasticNet 及其交叉验证变体）从探测器信号中恢复未知光谱。
- **噪声模拟** &mdash; 添加高斯噪声以模拟真实测量条件。
- **可视化** &mdash; 基于 Plotly 的交互式图表，支持响应度曲线、入射光谱、响应热力图和重构光谱的展示。

详细的 API 参考文档请参见 [Introduction_CN.md](Introduction_CN.md)。

## 安装

```bash
pip install -e .
```

**环境要求：** Python >= 3.10

**依赖项：** `numpy`、`scipy`、`scikit-learn`、`pandas`、`plotly`、`numba`

## 使用方法

### 快速上手

```python
import numpy as np
from SpectrumReconstruction import SpectrumReconstructionSimulation
from SpectrumReconstruction import SpectrumReconstructionAdvance as SRAdvance
from SpectrumReconstruction import Utility as SRUtility

# 定义参数
bias_array = np.linspace(0e-9, 500e-9, 501)          # 偏置数组 [m]
wavelength_array = np.linspace(400e-9, 1800e-9, 1401) # 波长数组 [m]
sigma = 1e-9 / 2 / np.sqrt(2 * np.log(2))             # FWHM = 1nm
mu = np.linspace(400e-9, 1800e-9, 1401)               # 中心波长数组 [m]

# 初始化模拟
srs = SpectrumReconstructionSimulation(
    bias_array=bias_array,
    wavelength_array=wavelength_array,
    base_function_name="gaussian",
    sigma=sigma,
    mu=mu,
    photo_detector_bias_mode="normal_move",
    delta_lambda=30e-9,   # 响应度过渡区宽度
    e_g_ev=0.75,          # 带隙能量 [eV]
    eta=1.0               # 量子效率
)

# 查看响应矩阵
srs.response_mapping_figure.show()

# 构造待测未知光谱
spectrum = SRAdvance.SimulationSpectrum(
    wavelength_array=wavelength_array,
    spectrum_function=SRUtility.gaussian_spectrum_sum
)
spectrum.set_spectrum(
    _mu=np.array([800e-9, 1200e-9]),
    _sigma=np.array([20e-9, 30e-9]),
    _alpha=np.array([1.0, 0.8])
)

# 执行光谱重构
srs.reconstruct_spectrum(
    simulation_spectrum=spectrum,
    method='ElasticNet',
    add_gaussian_noise=True,
    noise_std_ratio=0.01,
    lambda_reg=0.15,
    alpha=0.5
)

# 查看重构结果
srs.reconstruction_spectrum_figure.show()
```

### 模块结构

```
SpectrumReconstruction
├── SpectrumReconstructionBasic      # 光谱重构核心类（线性回归）
├── SpectrumReconstructionSimulation # 端到端模拟的高层封装
├── SpectrumReconstructionAdvance    # 探测器与光谱建模
│   ├── IdealSemiconductorPhotoDetector   # 理想半导体光电探测器
│   ├── IncidentSpectrum                  # 入射光谱（高斯/黑体）
│   ├── SimulationSpectrum                # 自定义待测光谱
│   ├── simulate_response_matrix()        # 计算响应矩阵
│   └── simulate_unknown_response()       # 计算未知光谱响应
└── Utility                          # 物理函数库
    ├── blackbody()                       # 黑体辐射（普朗克定律）
    ├── gaussian()                        # 高斯函数
    ├── smooth_responsivity()             # 平滑响应度（Sigmoid过渡）
    ├── ideal_responsivity()              # 理想阶跃响应度
    ├── gaussian_spectrum_sum()           # 加权高斯光谱叠加
    └── blackbody_spectrum_sum()          # 加权黑体光谱叠加
```

### 重构方法

| 方法名 | 说明 |
|---|---|
| `normal` | 普通最小二乘法（无正则化） |
| `l1` | L1 正则化（基于 SLSQP 优化器） |
| `l2` / `Ridge` | L2 正则化（岭回归） |
| `Lasso` | Lasso 回归（scikit-learn） |
| `LassoCV` | Lasso 回归（交叉验证自动选择 alpha） |
| `ElasticNet` | 弹性网络（L1 + L2 混合正则化） |
| `ElasticNetCV` | 弹性网络（交叉验证自动选择参数） |

### 示例

- [eample.ipynb](eample.ipynb) &mdash; `SpectrumReconstructionBasic` 类的分步演示
- [Simulation.py](Simulation.py) &mdash; 完整的模拟脚本（含全谱段与可见盲对比）

## 注意事项

- 所有波长和长度参数在内部均使用 **米 (m)** 为单位。传入时需从纳米转换，例如 1000 nm 应写为 `1000e-9`。
- 带隙能量参数 `e_g_ev` 的单位为 **eV**。
- 核心物理函数（`blackbody`、`gaussian`、`smooth_responsivity`）使用 Numba `@njit` 加速，首次调用时会因 JIT 编译而略慢。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。