# SpectrumReconstruction API 参考文档

**黄辰宇**$^{*\dagger}$

$^*$中国科学院上海技术物理研究所，红外科学与技术全国重点实验室

$^\dagger$中国科学院大学

[TOC]

## 概述

函数库```SpectrumReconstruction```旨在模拟光谱重构过程，其主要包含了光谱重构的基础类```SpectrumReconstructionBasic```和光谱重构的模拟类```SpectrumReconstructionSimulation```。函数的主要类和函数的调用层级如下：

```
SpectrumReconstruction
├── Class: SpectrumReconstructionBasic
├── Class: SpectrumReconstructionSimulation
├── Submodule: SpectrumReconstructionBasic
│   └── Class: SpectrumReconstructionBasic
│       ├── Method: __init__
│       ├── Method: base_func
│       ├── Method: reconstruct_spectrum
│       └── Method: spectrum
├── Submodule: SpectrumReconstructionAdvance
│   ├── Class: IdealSemiconductorPhotoDetector
│   ├── Class: IncidentSpectrum
│   ├── Class: SimulationSpectrum
│   │   ├── Method: set_spectrum
│   │   └── Property: spectrum_figure
│   ├── Function: simulate_response_matrix
│   ├── Function: simulate_unknown_response
│   └── Function: change_visible_blind_cutoff_parameter
├── Submodule: SpectrumReconstructionSimulation
│   └── Class: SpectrumReconstructionSimulation
│       ├── Method: __init__
│       ├── Method: reconstruct_spectrum
│       ├── Property: response_mapping_figure
│       └── Property: reconstruction_spectrum_figure
└── Submodule: Utility
    ├── Function: blackbody
    ├── Function: gaussian
    ├── Function: ideal_responsivity
    ├── Function: smooth_responsivity
    ├── Function: smooth_responsivity_visible_blind
    ├── Function: gaussian_spectrum_sum
    └── Function: blackbody_spectrum_sum
```

## 类 ```SpectrumReconstructionBasic```

```SpectrumReconstructionBasic```为光谱重构的基础类，其为光谱重构的基础类，用于实现光谱重构的基础功能。

关于类```SpectrumReconstructionBasic```的示例请参照[example.ipynb](../examples/example.ipynb)。

## 类 ```SpectrumReconstructionSimulation```

```SpectrumReconstructionSimulation```旨在全流程模拟光谱重构，其为模拟光谱重构过程中抽象层级最高的类。

### 构造方法```__init__```

构造函数接受多个参数来初始化类实例，参数如下：

1. ```bias_array```: 光电探测器的光谱响应度偏移数组，类型为```numpy.ndarray```，模拟探测器在不同状态下的光谱偏移，单位为nm；
2. ```wavelength_array```：全局光谱数组，类型为```numpy.ndarray```，用于确定整个模拟过程的光谱范围和步长，单位为nm；
3. ```base_function_name```：基函数名称，类型为```str```，用于确定模拟过程中的基础函数，目前支持```'gaussian'```和```'blackbody'```两种基础函数；
4. ```sigma```：高斯函数的标准差，类型为```float```，用于确定高斯函数的形状，模拟波长FWHM为```sigma```的高斯光入射，单位为nm；
5. ```mu```：高斯函数的均值数组，类型为```numpy.ndarray```，用于确定训练过程中一系列高斯函数的中心波长，单位为nm；
6. ```black_body_temperature```：黑体温度数组，类型为```numpy.ndarray```，用于确定训练过程中一系列黑体函数的温度，单位为K；
7. ```photo_detector_bias_mode```：光电探测器偏移模式，类型为```str```，用于确定模拟过程中的光电探测器偏移模式，目前支持```'normal_move'```和```'increase_band_gap'```两种模式，```'normal_move'```指将光电探测器的光谱响应度直接按照```bias_array```进行整体向右偏移，```'increase_band_gap'```指将改变光电探测器的带隙，将截止波长按照```bias_array```向右偏移；
8. ```delta_lambda```：指光电探测器的平滑区间长度，类型为```float```，用于确定光电探测器的平滑过程，单位为nm，具体平滑过程详见[Advance Analyse.ipynb](Advance Analyse.ipynb)；
9. ```e_g_ev```：指光电探测器的带隙，类型为```float```，单位为eV，默认为0.75(对应波长为1653.33nm，大约为InGaAs探测器的截止波长)；
10. ```eta```：指光电探测器的量子效率，类型为```float```，默认为1；
11. ```visible_blind_cutoff```：指光电探测器的可见光截止波长，用于模拟探测器的可见盲特性，类型为```float```，单位为nm，默认为-1(负值对应不设置可见光截止波长)。

### 属性汇总

#### 基础输入属性

1. ```bias_array```：偏置数组，单位为m
2. ```wavelength_array```：波长数组，单位为m
3. ```base_function_name```：基函数名称，可以是"blackbody"或"gaussian"
4. ```photo_detector_base_function```：光电探测器基函数，默认为smooth_responsivity
5. ```photo_detector_bias_mode```：光电探测器偏置模式，可以是"normal_move"或"increase_band_gap"
6. ```delta_lambda```：过渡宽度，默认30nm
7. ```e_g_ev```：带隙能量，默认0.75eV
8. ```eta```：量子效率，默认1.0
9. ```visible_blind_cutoff```：可见光截止波长，默认-1m

#### 基于输入的```base_function_name```的条件属性

##### 黑体模式下```base_function_name = 'blackbody'```

1. ```black_body_temperature```：黑体温度数组，单位为K

##### 高斯模式下```base_function_name = 'gaussian'```

1. ```sigma```：高斯函数的标准差，单位为m
2. ```mu```：高斯函数的均值数组，单位为m

#### 计算生成的属性

1. ```photo_detector```：类```IdealSemiconductorPhotoDetector```的实例，用于模拟光电探测器
2. ```incident_spectrum```：类```IncidentSpectrum```的实例，用于模拟入射光谱
3. ```response_pivot```：响应矩阵，用于模拟入射光谱照射到光电探测器后的响应
4. ```response_melted```：经过数据格式调整后的响应矩数据记录
5. ```spectrum_reconstruction```：类```SpectrumReconstructionBasic```的实例，用于模拟光谱重构过程

### 属性方法(@property)

1.  ```response_mapping_figure```：返回类型为```plotly.exprese.imshow```的响应矩阵heatmap，用于展示光谱重构过程中的响应矩阵。
2. ```reconstruction_spectrum_figure```：返回类型为```plotly.exprese.line```的重构光谱折线图，用于展示光谱重构过程中的重构光谱。

### 功能方法```reconstruct_spectrum```

```reconstruct_spectrum```方法用于模拟光谱重构过程，参数如下：
1. ```simulation_spectrum```：类```SimulationSpectrum```的实例，用用于构造一个复杂光谱输入探测器进行重构过程；
2. ```method```: 重构方法 ('normal', 'l1', 'l2', 'ElasticNet')，默认为'normal'；
3. ```add_gaussian_noise```: 是否添加高斯噪声，默认为False；
4. ```noise_std_ratio```: 噪声标准差比例，默认为0.1；
5. ```pivot_pass_in_test_data```: 用于标注输入的测试数据的格式，根据格式不同来确定是否需要对输入的数据进行透视处理，默认为False。

返回值为解算出的光谱系数a。

## 类 ```IdealSemiconductorPhotoDetector```

```IdealSemiconductorPhotoDetector```用于模拟理想半导体光电探测器的光谱响应特性，可以根据不同的偏置模式计算探测器在各波长下的响应度。

### 构造方法```__init__```

构造函数接受多个参数来初始化类实例，参数如下：

1. ```bias_array```：偏置数组，类型为```numpy.ndarray```，单位为m，用于定义探测器在不同偏置状态下的光谱偏移量；
2. ```e_g_ev```：带隙能量，类型为```float```，单位为eV，用于确定探测器的截止波长；
3. ```eta```：量子效率，类型为```float```，默认为1.0；
4. ```delta_lambda```：过渡区宽度，类型为```float```，单位为m，默认为30nm(30e-9)，用于控制响应度在截止波长附近的平滑过渡；
5. ```base_function```：基础响应度函数，类型为```Callable```，默认为```smooth_responsivity```，用于计算探测器的基础响应度曲线；
6. ```wavelength```：波长数组，类型为```numpy.ndarray```，单位为m，默认为```np.linspace(0, 2.0e-6, 500)```；
7. ```visible_blind_cutoff```：可见光截止波长，类型为```float```，单位为m，默认为-1（负值表示不启用可见盲特性）；
8. ```bias_mode```：偏置模式，类型为```str```，支持```'normal_move'```、```'increase_band_gap'```和```'decrease_eta'```三种模式：
   - ```'normal_move'```：将探测器的光谱响应度整体沿波长轴偏移；
   - ```'increase_band_gap'```：通过改变带隙能量来偏移截止波长；
   - ```'decrease_eta'```：在改变带隙的同时降低量子效率，量子效率随偏置量增大而降低。

### 属性方法(@property)

1. ```responsivity```（cached_property）：返回类型为```numpy.ndarray```的二维响应度矩阵（波长×偏置），利用向量化计算同时得到所有偏置下的响应度，避免迭代拼接的开销；
2. ```_responsivity```：将响应度矩阵转换为长格式的```pandas.DataFrame```，包含```wavelength```、```bias```和```responsivity```三列。

### 功能方法

1. ```responsivity_figure_show```：返回类型为```plotly.express.line```的折线图，展示不同偏置值下探测器的响应度随波长的变化关系。

## 类 ```IncidentSpectrum```

```IncidentSpectrum```用于模拟入射光谱，支持高斯光谱和黑体辐射光谱两种模式。

### 构造方法```__init__```

构造函数根据基函数类型接受不同参数：

1. ```wavelength```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```base_function_name```：基函数名称，类型为```str```，支持```'gaussian'```和```'blackbody'```两种模式；

**高斯模式下额外参数：**

3. ```sigma```：高斯函数的标准差，类型为```float```，单位为m，默认为1e-9；
4. ```mu```：高斯函数的中心波长数组，类型为```numpy.ndarray```，单位为m。

**黑体模式下额外参数：**

3. ```T```：黑体温度数组，类型为```numpy.ndarray```，单位为K。

### 属性方法(@property)

1. ```spectrum```：返回类型为```numpy.ndarray```的二维光谱矩阵（波长×参数），利用广播机制同时计算所有参数对应的光谱；
2. ```_spectrum```：将光谱矩阵转换为长格式的```pandas.DataFrame```，包含```wavelength```、基函数参数和```spectrum```三列。

### 功能方法

1. ```spectrum_figure_show```：返回类型为```plotly.express.line```的折线图，展示不同参数下入射光谱随波长的变化关系。

## 类 ```SimulationSpectrum```

```SimulationSpectrum```用于构造任意自定义光谱，作为光谱重构过程中的待测输入光谱。与```IncidentSpectrum```不同，该类允许用户通过自定义函数灵活定义光谱形状。

### 构造方法```__init__```

1. ```wavelength_array```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```spectrum_function```：光谱函数，类型为```Callable```，接受波长数组作为第一个参数，返回对应的光谱强度数组。

### 功能方法

1. ```set_spectrum(**kwargs)```：调用```spectrum_function```生成光谱并存储，```**kwargs```将被传递给光谱函数。返回值为生成的光谱数组（```numpy.ndarray```）。

### 属性方法(@property)

1. ```spectrum```：返回类型为```numpy.ndarray```的光谱数组。若未调用```set_spectrum```则抛出```ValueError```；
2. ```spectrum_figure```：返回类型为```plotly.express.line```的光谱折线图。

## 函数 ```simulate_response_matrix```

```simulate_response_matrix```用于计算光电探测器对入射光谱的响应矩阵，即将探测器的响应度矩阵与入射光谱矩阵进行矩阵乘法。

**参数：**

1. ```photodetector```：类```IdealSemiconductorPhotoDetector```的实例；
2. ```incident_spectrum```：类```IncidentSpectrum```的实例。

**返回值：** ```numpy.ndarray```，响应矩阵，行对应偏置值，列对应入射光谱参数（mu或T）。

**注意：** 要求```photodetector```和```incident_spectrum```的波长数组完全一致，否则抛出```ValueError```。

## 函数 ```simulate_unknown_response```

```simulate_unknown_response```用于模拟光电探测器对未知光谱的响应，可选添加高斯噪声以模拟实际测量中的噪声干扰。

**参数：**

1. ```photodetector```：类```IdealSemiconductorPhotoDetector```的实例；
2. ```unknown_spectrum```：类```SimulationSpectrum```的实例；
3. ```add_gaussian_noise```：是否添加高斯噪声，类型为```bool```，默认为```False```；
4. ```noise_std_ratio```：噪声标准差与响应均值的比值，类型为```float```，默认为0.01。

**返回值：** ```numpy.ndarray```，探测器对未知光谱的响应向量。

**注意：** 要求```photodetector```和```unknown_spectrum```的波长数组完全一致，否则抛出```ValueError```。

## 函数 ```blackbody```

```blackbody```计算黑体辐射的光谱辐射强度（普朗克定律），采用对数空间计算以提高数值稳定性，并使用```@njit```加速。

$$B(\lambda, T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc/(\lambda k T)} - 1}$$

**参数：**

1. ```lambda_```：波长，类型为```float```或```numpy.ndarray```，单位为m；
2. ```t```：温度，类型为```float```或```numpy.ndarray```，单位为K。

**返回值：** ```float```或```numpy.ndarray```，光谱辐射强度，单位为W/(m²·m)。支持标量与数组的广播运算。

## 函数 ```gaussian```

```gaussian```计算高斯函数值，使用```@njit```加速。

$$G(\lambda, \mu, \sigma) = \exp\left(-\frac{(\lambda - \mu)^2}{2\sigma^2}\right)$$

**参数：**

1. ```lambda_```：自变量（波长），类型为```float```或```numpy.ndarray```，单位为m；
2. ```mu```：均值（中心波长），类型为```float```或```numpy.ndarray```，单位为m；
3. ```sigma```：标准差，类型为```float```，单位为m。

**返回值：** ```float```或```numpy.ndarray```，高斯函数值（未归一化，峰值为1）。支持标量与数组的广播运算。

## 函数 ```ideal_responsivity```

```ideal_responsivity```计算理想半导体光电探测器的响应度，在截止波长以下呈线性增长，截止波长以上响应度为零。

$$R(\lambda) = \begin{cases} \eta \cdot \frac{q\lambda}{hc} & \lambda \leq \lambda_g \\ 0 & \lambda > \lambda_g \end{cases}$$

其中$\lambda_g = hc/E_g$为截止波长。

**参数：**

1. ```lambda_```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```e_g```：带隙能量，类型为```float```，单位为J；
3. ```eta```：量子效率，类型为```float```，默认为1.0。

**返回值：** ```numpy.ndarray```，响应度数组，单位为A/W。

## 函数 ```smooth_responsivity```

```smooth_responsivity```计算平滑响应度，相比```ideal_responsivity```在截止波长附近采用Sigmoid函数实现平滑过渡，更接近实际探测器的响应特性。使用```@njit```加速。

$$R(\lambda) = \eta \cdot \frac{q\lambda}{hc} \cdot \frac{1}{1 + \exp\left(\frac{\lambda - \lambda_g}{\Delta\lambda}\right)}$$

其中$\lambda_g = hc/E_g$为截止波长，$\Delta\lambda$为过渡区宽度。

**参数：**

1. ```lambda_```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```e_g```：带隙能量，类型为```float```，单位为J；
3. ```delta_lambda```：过渡区宽度，类型为```float```，单位为m，默认为30nm(30e-9)；
4. ```eta```：量子效率，类型为```float```，默认为1.0。

**返回值：** ```numpy.ndarray```，平滑响应度数组，单位为A/W。

## 函数 ```smooth_responsivity_visible_blind```

已废弃，不再使用。其功能已被```IdealSemiconductorPhotoDetector```类中的```visible_blind_cutoff```参数替代。

## 函数 ```gaussian_spectrum_sum```

```gaussian_spectrum_sum```计算多个加权高斯函数的叠加光谱，即$S(\lambda) = \sum_i \alpha_i \cdot G(\lambda, \mu_i, \sigma_i)$。

**参数：**

1. ```_wavelength```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```_mu```：各高斯函数的中心波长数组，类型为```numpy.ndarray```，单位为m；
3. ```_sigma```：高斯函数的标准差，类型为```float```或```numpy.ndarray```，单位为m。若为```float```则所有高斯函数共享同一标准差，若为```numpy.ndarray```则各高斯函数使用独立标准差；
4. ```_alpha```：各高斯函数的权重系数数组，类型为```numpy.ndarray```。

**返回值：** ```numpy.ndarray```，叠加后的光谱数组。常用于配合```SimulationSpectrum```构造复杂的待测光谱。

## 函数 ```blackbody_spectrum_sum```

```blackbody_spectrum_sum```计算多个加权黑体辐射光谱的叠加，即$S(\lambda) = \sum_i \alpha_i \cdot B(\lambda, T_i)$。

**参数：**

1. ```_wavelength```：波长数组，类型为```numpy.ndarray```，单位为m；
2. ```_T```：各黑体的温度数组，类型为```numpy.ndarray```，单位为K；
3. ```_alpha```：各黑体辐射的权重系数数组，类型为```numpy.ndarray```。

**返回值：** ```numpy.ndarray```，叠加后的光谱数组。常用于配合```SimulationSpectrum```构造复杂的待测光谱。
