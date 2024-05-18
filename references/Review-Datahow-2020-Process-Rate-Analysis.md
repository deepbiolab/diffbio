该工作提出通过对**特定速率进行多变量数据分析**和建模，**揭示了底层生物系统行为**的深入见解，这是通过分析代谢物浓度所无法获得的。



## Introduction

- **MVDA 应用及其影响**：
  - MVDA 有助于检测过程性能的变化和变异，但其分析结果受过程操作中的有意变化或接种反应器时的小变化影响较大。

- **Gnoth 等人的发现**：
  - 小的初始生物量浓度变化（接种后）在最佳生产条件下可能导致显著的过程性能变化，特别是最终滴度浓度的变化。
  - 结论：在最大生产率下运行的培养过程在经典意义上不能被认为是稳健的，因为最终浓度可能会显著变化。

- **底层生物生产系统的稳定性**：
  - 观察到的浓度变化可能是由于过程修改引起的“人为现象”，而非生物变化。
  
- **代谢建模的挑战**：
  - 代谢建模（如代谢流分析、流量平衡分析）可以阐明生物系统的变化，**但通常需要测量更多的浓度**，并且**模型开发复杂且耗时**。

- **MVDA 的实用性和局限性**：
  - 鉴于资源和时间限制，MVDA 是一种实用工具，能够突出数据差异。
  - **但目前尚不清楚这些差异是源于生物系统本身还是过程操作**。

### 1.1 浓度变化可能是与过程修改相关的“人为现象”

至少存在四种情景，在这些情景下，**浓度剖面的比较分析无法提供**对系统行为差异的代表性洞察，即过程监控、可重复性分析、培养基开发/优化和放大（图1）。

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.04.52.png" alt="CleanShot 2024-05-16 at 11.04.52" style="zoom:50%;" />

考虑以下基于简单物料平衡的示例，

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.03.56.png" alt="CleanShot 2024-05-16 at 11.03.56" style="zoom:50%;" />

描述了补料批操作理想混合生物反应器中体积（V）、生物量（X）和底物（S）浓度随时间的演变：

##### 1.1.1 过程监控/可重复性分析

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.10.13.png" alt="CleanShot 2024-05-16 at 11.10.13" style="zoom:33%;" />

在典型的指数增长阶段，**初始生物量浓度的小变化会被放大，导致阶段末的生物量显著不同**，从而导致不同的底物剖面（图1A）。这是否意味着过程不稳健或不符合规范？尽管可以减少初始生物量浓度的波动，但我们认为，如果下游过程（特别是色谱纯化步骤）不受影响（因为更多的生物量可能增加需要去除的杂质量），从生物学角度来看，所有剖面实际上是可比的，因为特定速率没有表现出差异/变异。

##### 1.1.2 培养基开发

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.09.43.png" alt="CleanShot 2024-05-16 at 11.09.43" style="zoom:33%;" />

最简单的比较不同培养基的方法是通过改变不同实验的初始底物浓度进行模拟（图1B）。同时，我们可以假设初始生物量浓度存在差异，但为了简化，不包括这些波动。比较使用不同培养基进行的不同实验中底物浓度随时间的演变，几乎没有提供关于底层过程性能的洞察，因为这些变化仅仅是由于培养基浓度的变化。实际上，需要评估多个浓度剖面（即喂料速度和喂料培养基中的浓度）。

##### 1.1.3 放大/缩小和喂料速率优化研究

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.10.28.png" alt="CleanShot 2024-05-16 at 11.10.28" style="zoom:33%;" />

放大/缩小是生物加工中最具挑战性的问题之一。通常，通过保持限制因素（如质量传递速率或无量纲数，如单位体积功率）在不同尺度上相同来完成放大/缩小（图1C）。在图1C展示的放大情景中，**体积在不同尺度上变化**，但**喂料速率的调整被认为是不必要的**，因为底物浓度不被认为是限制因素。然而，可以观察到浓度剖面演变的影响，特别是在**高浓度下**（即，项 −u/V·S 会增加），这会导致非常不同的轨迹。**观察到的变化可以归因于稀释的变化，但底层生物系统行为相同**。

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.12.22.png" alt="CleanShot 2024-05-16 at 11.12.22" style="zoom:33%;" />

相反，图1D展示的放大情景在过程中浓度变化非常小，但底层生物系统在不同尺度上表现不同。

##### 1.1.4 一般结论
| 场景示例                    | 有意过程变化的影响                                           | 固有变化的影响                                 | 结果及解释                                                   |
| --------------------------- | ------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| 图1A: 过程监控/可重复性分析 | 初始生物量浓度的小变化会被放大，导致阶段末的生物量显著不同，进而影响底物剖面 | 底层生物系统生产模式可能未变，变化是“人为现象” | 不影响下游操作时，**剖面从生物学角度是可比的**；可能导致实验被拒绝作为可重复批次的一部分 |
| 图1B: 培养基开发            | 通过改变初始底物浓度模拟不同培养基的影响                     | **变化仅由于培养基浓度的差异，未涉及代谢变化** | **洞察有限，需要评估多个浓度剖面**；可能导致不必要的实验生成 |
| 图1C: 放大研究              | 不同尺度上的体积变化但喂料速率未调整，导致浓度剖面显著变化   | 底层生物系统行为相同                           | **变化归因于稀释的变化，实际系统行为一致**                   |
| 图1D: 放大研究              | 浓度变化较小但不同尺度上系统行为不同                         | **底层生物系统表现不同**                       | 需要验证喂料剖面的修改，拒绝缩小模型                         |

我们**主张在特定速率空间**（即特定吸收和生产速率）**而非浓度空间中应用 MVDA 和模型开发**。通过两个工业示例（HEK 培养基比较研究和 CHO 放大研究），展示了将重点转移到速率空间的影响。这种方法展示了数据转换（可理解为一种“特征工程”形式）如何帮助更好地理解和建模系统。



### 2.1 培养基比较研究

**实验目的**：

- 本研究旨在识别影响 HEK293 细胞培养存活细胞密度的代谢物。
- 具体来说，分析重点是调查培养基家族 B 和 C 的性能，并将其与参考培养基 A 进行比较。
- 简言之，我们希望了解参考培养基 A 与培养基 B 和 C 之间过程差异的来源。



#### 2.1.1 数据生成

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.24.24.png" alt="CleanShot 2024-05-16 at 11.24.24" style="zoom:50%;" />

| **实验步骤**   | **描述**                                                     |
| -------------- | ------------------------------------------------------------ |
| **设备和时间** | 在 24 个 Ambr250 设备中进行 8 天的 HEK293 批量培养           |
| **研究条件**   | 13 组条件，涉及 3 种培养基家族和预培养池                     |
| **取样时间点** | 第 0、1、2、3、5 和 8 天                                     |
| **分析内容**   | 20 种氨基酸（UPLC）、葡萄糖、铵、甘油、乳酸、尿素、钙、镁（Cedex-Bio） |
| **性能指标**   | 存活细胞密度                                                 |



#### 2.1.2 数据转换

**背景**：

- **测量代谢物浓度时会有测量误差**。
- 从浓度测量值估算特定速率是一个病态的逆问题，**测量误差可能被放大，导致特定速率的不确定性范围很大**。
- 为了量化这种情况的程度，使用基于蒙特卡洛方法的抽样方法。

**方法步骤**：

1. 从正态或均匀分布中抽取一个随机值，使用该测量的标准偏差，针对每个测量浓度的时间点。将这些值添加到原始测量浓度值中。
2. 对生物量浓度执行相同操作，因为这些浓度将用于计算特定速率。
3. 使用实验部分描述的方法，通过分段三次插值（函数 csaps，Matlab 2016a）从修改后的浓度值中估算速率，以近似测量值（即 f(t, w) 是一个光滑的分段三次样条）。
4. 将速率值与先前的值组合并重复步骤（1）到（4），直到在不断增长的速率数据集上计算的速率标准偏差估计值收敛。

**假设和参数**：
- 对于每个浓度数据，假设变异系数为10%，但对于生物量测量，假设误差仅为5%。
- 假设最小实验误差为 0.1 mg L−1，最大为 10 mg L−1。
- 生成100个随机测量值，均匀分布在与该点相关的实验置信区间内。
- 使用这些随机生成的数据，为每个测量的代谢物构建一个样条。

**样条平滑参数**：
- 使用 MATLAB 的 csaps 函数定义浓度数据的样条，确定合适的平滑参数 p。
- 默认的平滑参数公式为：p = 1/(1+(h^3/60))，其中 h 是平均采样时间间隔。
- 使用默认平滑参数检查样条是否在测量的置信区间内，如果不在，增加平滑参数直到样条穿过所有置信区间。

**结果**：
- 计算每个100个随机样条的速率，图表展示了所有计算速率的平均值和相关的置信区间（±标准差误差）。

下面是使用 Python 和相关库（如 NumPy 和 SciPy）实现上述方法的示例代码：

```python
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# 示例数据
time_points = np.array([0, 1, 2, 3, 5, 8])
original_concentrations = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
biomass_concentrations = np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1])
n_samples = 100

# 测量误差
cv_concentration = 0.10
cv_biomass = 0.05

# 生成随机测量值
random_concentrations = np.array([np.random.normal(loc=val, scale=val * cv_concentration, size=n_samples) for val in original_concentrations])
random_biomass = np.array([np.random.normal(loc=val, scale=val * cv_biomass, size=n_samples) for val in biomass_concentrations])

# 使用分段三次插值（CubicSpline）构建样条
def generate_spline(time_points, values):
    return CubicSpline(time_points, values, bc_type='natural')

# 计算特定速率
def compute_specific_rate(concentration_spline, biomass_spline, time_points):
    concentration_rates = concentration_spline(time_points, 1)  # 一阶导数
    biomass_rates = biomass_spline(time_points, 1)  # 一阶导数
    specific_rates = concentration_rates / biomass_rates
    return specific_rates

# 进行蒙特卡洛模拟
all_specific_rates = []

for i in range(n_samples):
    concentration_spline = generate_spline(time_points, random_concentrations[:, i])
    biomass_spline = generate_spline(time_points, random_biomass[:, i])
    specific_rates = compute_specific_rate(concentration_spline, biomass_spline, time_points)
    all_specific_rates.append(specific_rates)

all_specific_rates = np.array(all_specific_rates)
mean_specific_rates = np.mean(all_specific_rates, axis=0)
std_specific_rates = np.std(all_specific_rates, axis=0)

# 绘图
plt.figure()
plt.plot(time_points, original_concentrations)
plt.xlabel('Time')
plt.ylabel('Concentrations')
plt.show();
plt.figure()
plt.errorbar(time_points, mean_specific_rates, yerr=std_specific_rates, fmt='-o')
plt.xlabel('Time')
plt.ylabel('Specific Rates')
plt.title('Mean Specific Rates with Confidence Intervals')
plt.show();

```

通过这种方法，**可以有效地量化测量误差对特定速率估算的影响**，确保数据分析的可靠性。



##### 图2：样条近似和特定速率

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.38.59.png" alt="CleanShot 2024-05-16 at 11.38.59" style="zoom:50%;" />

- **样条近似**：
  - 图2展示了样条近似和速率的代表性示例，说明了**浓度在测量的置信区间内均匀分布**，符合预期。
  - 样条近似的目的是生成平滑的浓度曲线，以便在浓度数据存在测量误差的情况下估算特定速率。

- **特定速率的置信区间**：
  - 估算的特定速率**在浓度剖面变化的区域显示出稍大的置信区间**，而在其他区域则更为紧凑。
  - 这种置信区间的变化反映了在蒙特卡洛采样过程中构建的一组样条曲线的曲率变化最小化的方法，同时考虑了实验测量误差。
  - 这意味着**在浓度快速变化的区域，特定速率的估计值不确定性更大**。

##### 图3：小提琴图中的特定速率数据分布

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 11.40.24.png" alt="CleanShot 2024-05-16 at 11.40.24" style="zoom:50%;" />

- **特定速率数据的分布**：
  - 这一观察结果表明，**尽管输入的浓度数据是均匀分布的，通过计算特定速率后，结果数据趋向于正态分布。**
  - 这种现象**是否具有普遍性尚不明确**，可能是个别案例特有的特征。



#### 2.1.3 PCA 分析的洞察

**主成分分析（PCA）**：
- 批次展开的浓度数据和速率（流量）数据分别进行了两次 PCA 分析。
- 数据在变量维度上进行自动缩放，使用三个潜在成分描述数据的变异，对于浓度数据为 66.15% ± 1%，对于速率数据为 52.42% ± 1.73%。
- 图4展示了浓度和速率数据的前三个潜在成分的得分图。

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 13.15.19.png" alt="CleanShot 2024-05-16 at 13.15.19" style="zoom:50%;" />

- **浓度数据**：
  - 比较不同培养基**在浓度的潜在空间**，发现 A(red)、B(blue) 和 C(green) 导致不同的培养，**培养基 B 和 C 与参考培养基 A 显著不同**。
- **速率数据**：
  - 在**速率的潜在空间**中，**参考培养基 A 的培养在某些条件下与其他培养基 B 和 C 的培养相当接近**。
  - 这些不同的行为表明，**尽管培养基在浓度上有所不同，但细胞代谢是相似的**，因此可以预期产品质量相似，但数量不一定相同（与浓度相关）。

**载荷分析**：

<img src="Review-Datahow-2020-Process-Rate-Analysis.assets/CleanShot 2024-05-16 at 13.19.55.png" alt="CleanShot 2024-05-16 at 13.19.55" style="zoom:100%;" />

- **浓度负载**：
  - 浓度负载提供了**每一天实验浓度主要差异的直接洞察**。可以看到，**随着时间的推移，负载的位置更加分散**，突显了第1天的浓度变化如何影响培养的演变，导致更多浓度的差异。
  - 特定化合物负载在几天内的演变不应单独分析，因为每一天负载的位置受其他化合物的影响。
- **速率负载**：
  - **速率负载表明化合物的吸收/分泌差异**，指示实验间代谢的差异。
  - 由于细胞来自同一个预培养池，它们在开始时代谢相似，只是某些速率发生变化，表示细胞对有意添加化合物的初始响应。
  - **随着时间的推移，速率变化变得更加显著，反映了细胞对暴露环境和培养基的适应。**
  - 有趣的是**，第3天和第5天乳酸和谷氨酰胺在速率负载中起到更重要的作用**，这些途径的吸收/分泌受条件和培养基变化的影响。

以下是使用 Python 和相关库（如 NumPy 和 scikit-learn）进行 PCA 分析的示例代码：

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 示例数据（需要替换为实际数据）
concentration_data = np.random.rand(100, 20)  # 假设100个样本，20个浓度变量
rate_data = np.random.rand(100, 20)  # 假设100个样本，20个速率变量

# 标准化数据
concentration_data_scaled = (concentration_data - np.mean(concentration_data, axis=0)) / np.std(concentration_data, axis=0)
rate_data_scaled = (rate_data - np.mean(rate_data, axis=0)) / np.std(rate_data, axis=0)

# PCA 分析
pca_concentration = PCA(n_components=3)
pca_rate = PCA(n_components=3)

pca_concentration.fit(concentration_data_scaled)
pca_rate.fit(rate_data_scaled)

# 得分图
scores_concentration = pca_concentration.transform(concentration_data_scaled)
scores_rate = pca_rate.transform(rate_data_scaled)

# 图示
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(scores_concentration[:, 0], scores_concentration[:, 1], c='blue', label='Concentration Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Score Plot - Concentration Data')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(scores_rate[:, 0], scores_rate[:, 1], c='red', label='Rate Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Score Plot - Rate Data')
plt.legend()

plt.show()
```

| **分析项目**     | **描述**                                                     |
| ---------------- | ------------------------------------------------------------ |
| **PCA 分析**     | 分别对批次展开的浓度和速率数据进行了 PCA 分析，三个潜在成分足以描述数据的主要变异。 |
| **浓度数据结果** | PCA 得分图显示，不同培养基 A、B、C 导致不同的培养，B 和 C 明显区别于 A。 |
| **速率数据结果** | 在速率的潜在空间中，参考培养基 A 的培养与 B 和 C 培养在某些条件下接近，表明代谢相似。 |
| **负载分析**     | **浓度负载**显示每天的主要**浓度差异**，**速率负载**显示化合物吸收/分泌的差异，**反映代谢差异。** |

通过这种分析，可以更好地理解不同培养基对细胞代谢和生长的影响，以及这些变化如何影响产品质量和数量。
