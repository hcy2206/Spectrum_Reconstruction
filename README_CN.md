<!-- SpectrumReconstruction -->

<!--suppress HtmlDeprecatedAttribute -->
<div align="center" style="text-align:center">
   <h1> SpectrumReconstruction </h1>
   <p>
      光谱重构与光电探测器模拟 Python 模块<br>
      <code><b> v0.2 </b></code>
   </p>
   <p>
      <img alt="GitHub Top Language" src="https://img.shields.io/github/languages/top/hcy2206/Spectrum_Reconstruction?label=Python">
      <img alt="GitHub License" src="https://img.shields.io/github/license/hcy2206/Spectrum_Reconstruction?label=License"/>
   </p>
   <p>
      <a href="README.md">English</a> | 中文
   </p>
</div>

## 简介

**SpectrumReconstruction** 是一个用于从经标定的光电探测器信号中重构光谱的 Python 软件包。它还提供光电探测器响应度建模和测量仿真工具，可用于评估测量噪声对重构光谱的影响。

本软件包是论文《End-to-end optoelectronic architecture for event-driven infrared spectroscopy》的配套软件。

<p>
      <b>论文作者</b><br>
      黄辰宇<sup>1,2,4</sup>、邓铭鑫<sup>3,4</sup>、夏辉<sup>1,2,4</sup>、许航瑀<sup>1,2*</sup>、王海露<sup>1,2</sup>、黄子华<sup>1,2</sup>、王坤宇<sup>1</sup>、付晓<sup>1,2</sup>、王义楠<sup>3*</sup>、李清江<sup>3</sup>、胡伟达<sup>1,2*</sup><br>
      <sup>1</sup> 中国科学院上海技术物理研究所 红外科学与技术全国重点实验室<br>
      <sup>2</sup> 中国科学院大学<br>
      <sup>3</sup> 国防科技大学电子科学学院<br>
      <sup>4</sup> 黄辰宇、邓铭鑫 和 夏辉 对本工作贡献相同。<br>
      <sup>*</sup> 通讯作者：许航瑀 (<a href="mailto:xuhangyu@mail.sitp.ac.cn">xuhangyu@mail.sitp.ac.cn</a>), 王义楠 (<a href="mailto:wangyinan@nudt.edu.cn">wangyinan@nudt.edu.cn</a>), 胡伟达 (<a href="mailto:wdhu@mail.sitp.ac.cn">wdhu@mail.sitp.ac.cn</a>)
   </p>

## 主要功能

- **光电探测器建模** &mdash; 模拟理想半导体光电探测器的光谱响应度，支持可调带隙、量子效率、多种偏置模式（`normal_move`、`increase_band_gap`、`decrease_eta`）及可见盲特性。
- **入射光谱生成** &mdash; 使用高斯函数或黑体辐射模型生成训练光谱。
- **响应矩阵计算** &mdash; 计算探测器对一组已知入射光谱的响应矩阵。
- **光谱重构** &mdash; 通过线性回归及可选的正则化方法（OLS、Lasso、Ridge、ElasticNet 及其交叉验证变体）从探测器信号中恢复未知光谱。
- **噪声模拟** &mdash; 添加高斯噪声以模拟真实测量条件。
- **可视化** &mdash; 基于 Plotly 的交互式图表，支持响应度曲线、入射光谱、响应热力图和重构光谱的展示。

参见[模块架构](docs/Architecture_CN.md)和 [API 参考](docs/Module_and_API_Reference_CN.md)。

## 安装

```bash
git clone https://github.com/hcy2206/Spectrum_Reconstruction.git
cd Spectrum_Reconstruction
pip install -e .
```

**环境要求：** Python >= 3.10

**依赖项：** `numpy>=2.0.1,<2.2.0`、`scipy`、`scikit-learn`、`pandas`、`plotly`、`numba`

## 使用方法

### 快速上手

#### 从探测器信号重构光谱

在 Jupyter 中以 `examples/` 为工作目录，按顺序运行[重构示例 Notebook](examples/example.ipynb)。[标定数据](examples/data_training_example.csv)记录了不同黑体温度和偏置下的探测器电流，[测试数据](examples/data_testing_example.csv)提供待重构光源的测量记录。每行对应一次测量，三列分别为偏置（`V_bias/mV`）、电流（`Current`）和温度（`Temperature`）。示例使用黑体基函数和 L1 正则化重构光谱，最后绘出归一化光谱强度随波长的变化。

#### 模拟探测器并重构双峰光谱

以下示例生成峰位在 800 nm 和 1200 nm 的输入光谱，模拟含高斯噪声的探测器信号，再用 ElasticNet 重构光谱。波长以米传入，例如 800 nm 写作 `800e-9`。最后的叠加图可用于比较输入谱与重构谱的峰位和峰形。

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
fig = srs.reconstruction_spectrum_figure
fig.update_traces(name="Reconstructed", showlegend=True)
fig.add_scatter(x=wavelength_array, y=spectrum.spectrum, mode="lines", name="Input")
fig.update_xaxes(title_text="Wavelength (m)")
fig.show()
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

## 注意事项

- 所有波长和长度参数在内部均使用 **米 (m)** 为单位。传入时需从纳米转换，例如 1000 nm 应写为 `1000e-9`。
- 带隙能量参数 `e_g_ev` 的单位为 **eV**。
- 核心物理函数（`blackbody`、`gaussian`、`smooth_responsivity`）使用 Numba `@njit` 加速，首次调用时会因 JIT 编译而略慢。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE) 文件。
