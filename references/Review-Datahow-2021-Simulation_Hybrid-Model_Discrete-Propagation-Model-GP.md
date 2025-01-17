### **Abstract**

1. **实验数据的单次使用问题**：目前的生化过程开发过程中，生成的数据通常只用于当下的决策，而没有充分利用这些数据来为未来的过程做预测。这意味着每次开发新过程都需要进行大量的实验。

2. **跨产品数据学习的挑战**：不同产品的过程行为存在差异，只有一部分产品行为相似。因此，要在多个产品的数据上进行有效学习，需要一个能够合理表示产品身份的方法。

3. **嵌入向量的提出**：论文提出用嵌入向量来表示产品身份，这些**嵌入向量作为高斯过程回归模型的输入**。嵌入向量可以从过程数据中学习，并且能够捕捉产品之间的相似性。

4. **性能比较**：研究表明，与传统的one-hot编码方法相比，使用嵌入向量的方法在模拟跨产品学习任务中性能有所提升。

5. **实验减少的潜力**：使用该方法，可以显著减少实验室实验的数量，从而提高效率并降低成本。

这篇论文的核心是通过引入嵌入向量来改进跨产品数据的学习，使得可以更有效地利用已有数据来预测新过程，减少实验需求。



### Introduction

1. **生产挑战**：治疗性蛋白质和疫苗的生产需要非常精确地控制质量，同时要经济可行，并且保护环境、健康和安全。
   
2. **开发成本高**：这些工艺的开发过程非常耗时、不可靠且成本高，因为每个产品的工艺设计都需要个性化。
   
3. **高通量平台的兴起**：微型高通量平台的出现使得可以进行成本和时间高效的并行实验。
   
4. **数据分析瓶颈**：随着高通量技术的兴起，数据收集和分析以及设计具有信息量的平行实验成为新的瓶颈。
   
5. **现有方法的局限性**：现有的统计实验设计方法在很大程度上忽视了先验知识，尤其是细胞系变化的情况下。
   
6. **知识转移的困难**：水平知识转移仍然有限，每种新产品都需要进行大量实验来理解其过程行为。
   
7. **新方法的需求**：需要新的方法从不同产品生成的过程数据的联合分析中获取信息，并能够在生成少量数据时预测新产品的过程行为。



### 多产品跨Process数据分析的三种方法：

目前有三种方法用于分析多个产品的跨过程数据，如下所述：

1. **一次性编码（One hot encoding）**：
   - **方法**：将每个数据点的产品身份处理为一个分类过程变量，并将其转换为一次性（虚拟）变量【20†source】。
   - **优点**：在分析中区分不同产品的数据可能是高效的。
   - **局限性**：这种编码方式不携带不同产品之间相似性的信息，学习成果的转移性有限。此外，无法筛选和提取对新产品有用的信息。

2. **缩放（Scaling）**：
   - **方法**：通过缩放来提高不同产品的过程数据的可比性【20†source】【21†source】。例如，可以在过程中使用缩放来“标准化”绝对滴度的差异或了解不同设定点周围pH变化的影响。
   - **优点**：可以提高数据的可比性。
   - **局限性**：如果缩放未能使数据更具可比性，那么联合分析可能提供的洞察与单独分析每个产品的数据相差无几。

3. **手工制作的知识驱动特征创建（Handcrafted, knowledge-based feature creation）**：
   - **方法**：将描述产品特性的特征整合到分析中，从而实现跨产品数据分析。为此，已经生成并使用了分子描述符，产生了定量构效关系（QSAR）模型、定量序列-活性模型（QSAM）或卷积图网络【21†source】。
   - **优点**：可以在跨产品数据分析中利用产品特性。
   - **局限性**：捕捉药物物质的特性并从生成的大量特征中选择感兴趣的特征并不容易。

这些方法各有优缺点，一次性编码和缩放在处理和比较数据方面可能相对简单，但它们在捕捉产品相似性和提取对新产品有用信息方面存在不足。而手工制作的特征创建虽然能够更好地进行跨产品分析，但其复杂性和高要求使其难以广泛应用。论文提出需要新的方法来克服这些局限，以更好地利用已有数据，减少实验需求。



### 本研究的创新

1. **产品嵌入与高斯过程模型**：
   - **嵌入概念**：受自然语言处理中用高维向量表示单词的启发，我们将产品表示为嵌入向量。嵌入可以将具有相似意义的实体靠近，并展示加法组合性。
   - **高斯过程回归的优势**：相比于神经网络，高斯过程回归对**数据的需求较低**，并且**可以评估预测的不确定性**，这使其在生物工艺建模中特别有用。
   - **目标**：将嵌入向量的适用性扩展到高斯过程回归中，以便更好地捕捉产品之间的相似性。

2. **模型结构**：
   - **描述过程操作变化**：使用物料平衡明确描述来自过程操作的变化。
   - **混合半参数模型**：结合嵌入向量与高斯过程回归，构成一个混合半参数模型，用于跨产品的数据建模。

### 研究步骤：

1. **学习产品嵌入的方法**：
   - 描述了如何使用高斯过程模型学习产品嵌入，并将其集成到混合半参数过程中。
   
2. **数学见解**：
   - 阐明了传统one-hot向量在高斯过程回归中的局限性，并展示了学习嵌入如何提供关于产品相似性的洞察。

3. **模拟案例研究**：
   - 通过模拟案例研究，严格评估所提出的方法。
   
4. **总结研究结果**：
   - 最后，总结研究结果并讨论其潜在应用。





### Dataset

在过程开发中，目标是找到能够在短时间内持续生产高质量滴度（产品浓度）的工艺条件$\mathbf{E}$，例如温度、初始浓度等。为了有效支持新产品的过程开发，我们需要一个模型能够预测未知过程条件$\mathbf{E}'$下的系统行为，以便用于过程优化。

系统行为表示为一个浓度矩阵$\mathbf{M}' \in \mathbb{R}^{N_{\text{time}} \times N_Q}$，其中：

- $N_Q$是相关量的数量，如下所示：
$$
N_Q = 6 \quad \\
(\text{include: Viable Cell Density (VCD), Glucose, Glutamine, Ammonia, Lactate 和 Titer})
$$

- $N_{\text{time}}$是时间点的数量：
$$
N_{\text{time}} = 14
$$

为了创建这样的模型，可以使用过去为不同产品$N_{\text{prods}}$生成的$N_E$次实验运行的数据。对于每次运行$i \in \{1, \ldots, N_E\}$，已知过程条件$\mathbf{E}_i$和产品$p_i \in \{1, \ldots, N_{\text{prods}}\}$以及浓度矩阵$\mathbf{M}_i \in \mathbb{R}^{N_{\text{time}} \times N_Q}$。

定义：

- **浓度矩阵**$\mathbf{M}_i$：
$$
\mathbf{M}_i = \begin{bmatrix}
m_{i,1,1} & m_{i,1,2} & \cdots & m_{i,1,N_Q} \\
m_{i,2,1} & m_{i,2,2} & \cdots & m_{i,2,N_Q} \\
\vdots & \vdots & \ddots & \vdots \\
m_{i,N_{\text{time}},1} & m_{i,N_{\text{time}},2} & \cdots & m_{i,N_{\text{time}},N_Q} \\
\end{bmatrix}
$$

- **过程条件**$\mathbf{E}_i$：

  表示每次实验运行的具体条件，如温度、初始浓度等。

- **产品身份**$p_i$：

  表示每个数据点所属的产品。

目标是通过模型预测在新的过程条件$\mathbf{E}'$下系统的行为，表示为：

$$
\mathbf{M}' = f(\mathbf{E}')
$$

模型的优化目标是找到最优的过程条件$\mathbf{E}^*$，使得高质量滴度$\mathbf{T}$最大化：
$$
\mathbf{E}^* = \arg\max_{\mathbf{E}} \mathbf{T}(\mathbf{E})
$$

通过使用嵌入向量和高斯过程回归，可以更好地捕捉不同产品间的相似性，提高模型的预测精度和可靠性。这种方法不仅能减少实验需求，还能加速新产品的开发过程。



## 混合回归模型

### 显式建模补料

动态浓度演变的建模受到理想混合反应器的动态物料平衡的启发。具体地，通过时间离散化，我们对训练集中的每个实验$i \in \{1, \ldots, N_E\}$​​ 中:

#### 训练

##### OUTPUT

###### 无补料:

前后两个等间隔时刻测量之间的浓度变化斜率进行近似，如下所示：

- Consumption variable

$$
\frac{dM^i}{dt}=-k(\mathrm{M}^i)\\
-k(\mathrm{M}^i) = \frac{dM^i}{dt}  \\
y_{t}^i := k(\mathrm{M}^i) = -\frac{dM^i}{dt}  \\
\text{where }\frac{dM^i}{dt} ≈ \frac{\Delta M^i_t}{\Delta t} = \bold{-} \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t}
$$

- Increation variable
$$
\frac{dM^i}{dt}=k(\mathrm{M}^i)\\
k(\mathrm{M}^i) = \frac{dM^i}{dt} \\
y_{t}^i := k(\mathrm{M}^i) = \frac{dM^i}{dt}  \\
\text{where }\frac{dM^i}{dt} ≈ \frac{\Delta M^i_t}{\Delta t} = \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t}
$$


  其中:

  - $M_{t}^i \in \mathbb{R}^{N_Q}$是在时间$t$的测量向量，
  - $\Delta t$​​ 是两个测量之间的时间间隔。
  - **$\Delta F_{t+{\Delta t}}$是离线检测完后,立马发生的补料动作,从数据的形式上,其时间与当前时刻$t$​​的离线检测状态变量位于相同行**, 注意此处为质量, 不是浓度
  - $-k(\mathrm{M}^i)$表示浓度$i$处于消耗状态
  - $k(\mathrm{M}^i)$表示浓度$i$处于增加状态
  - $y_{t}^i$是表示浓度$i$​的速率,用于建模中的标签


$$
\frac{dM^i}{dt}=-k(\mathrm{M}^i) + \text{feed rate} \\
\frac{dM^i}{dt}=-k(\mathrm{M}^i) + \frac{\Delta F_{t+{\Delta t}}}{\Delta t} \\
-k(\mathrm{M}^i) = \frac{dM^i}{dt} - \frac{\Delta F_{t+{\Delta t}}}{\Delta t}  \\
y_{t}^i := k(\mathrm{M}^i) = -\frac{dM^i}{dt} + \frac{\Delta F_{t+{\Delta t}}}{\Delta t}  \\
\text{where }\frac{dM^i}{dt} ≈ \frac{\Delta M^i_t}{\Delta t} = \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t} \\
y^i_t = \frac{M_{t+{\Delta{t}}} - M_t}{\Delta t} - \frac{\Delta F_{t+{\Delta{t}}}}{\Delta t}
$$




###### 有补料:

以某个浓度$i$的消耗为例:
$$
\frac{dM^i}{dt}=-k(\mathrm{M}^i) + \text{feed rate} \\
\frac{dM^i}{dt}=-k(\mathrm{M}^i) + \frac{\Delta F_{t+{\Delta t}}}{V_t \cdot \Delta t} \\
-k(\mathrm{M}^i) = \frac{dM^i}{dt} - \frac{\Delta F_{t+{\Delta t}}}{V_t \cdot \Delta t}  \\
y_{t}^i := k(\mathrm{M}^i) = -\frac{dM^i}{dt} + \frac{\Delta F_{t+{\Delta t}}}{V_t \cdot \Delta t}  \\
\text{where }\frac{dM^i}{dt} ≈ \frac{\Delta M^i_t}{\Delta t} = \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t} \\
y^i_t = \frac{M_{t+{\Delta{t}}} \cdot V_{t+{\Delta{t}}} - M_t \cdot V_t}{V_t \cdot \Delta t} - \frac{\Delta F_{t+{\Delta{t}}}}{V_t \cdot \Delta t}
$$


  其中:

  - $\Delta F_{t+{\Delta{t}}} \in \mathbb{R}^{N_Q}$是由于在时间$t$进料引起的质量变化向量, 在我们的实验中，我们有葡萄糖和谷氨酰胺进料，因此向量$\Delta F$只有在对应于这两种物质的位置上有非零条目
  - $V_t \in \mathbb{R}^+$​ 是离散时间点的体积
  - 补料$\Delta F_{t+{\Delta{t}}}$对浓度的影响体现在时刻$t+{\Delta{t}}$的浓度测量中，但在训练模型时，我们希望将这种影响分离出来，以更准确地捕捉系统的动力学特性。

  

##### INPUT

通过将步骤开始时的浓度$M_{t}^i$与实验设计中获得的额外特征（如反应器温度和pH值）$\text{ExtraFeatures}(E_i) \in \mathbb{R}^k$连接起来，构建特征向量$x_{t}^i$​​。对于训练集中的所有实验，这样做后我们得到矩阵：
$$
X \in \mathbb{R}^{(N_E \cdot N_{\text{time}}) \times (N_Q + k)} \\
Y \in \mathbb{R}^{(N_E \cdot N_{\text{time}}) \times N_Q}
$$


##### Modelling

使用$X, Y$, 训练任意回归模型$\Phi : \mathbb{R}^{N_Q + k} \rightarrow \mathbb{R}^{N_Q}$，该模型描述了反应器类比中的反应动力学。



#### 预测

##### 无补料

对于未知实验条件$E'$的预测如下所示：
$$
M'_0 = \text{InitialConcentration}(E') \\
M'_{t+{\Delta{t}}} = M'_t + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot \Delta t
$$

##### 有补料

- 质量版本

$$
M'_{t+{\Delta{t}}} \cdot V_{t+{\Delta{t}}} = [M'_t + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot \Delta t] \cdot V_t + \Delta F_{t+{\Delta{t}}} \\
$$
- 浓度版本

$$
M'_{t+{\Delta{t}}} = \left( M'_t  + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot \Delta{t} \right) \cdot \frac{V_t}{V_{t+{\Delta{t}}}} + \frac{\Delta F_{t+{\Delta{t}}}}{V_{t+{\Delta{t}}}} \\
	    M'_{t+{\Delta{t}}} = \left( M'_t \cdot \frac{1}{\Delta{t}}  + \Phi((M'_t, \text{ExtraFeatures}(E')))  \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \cdot V_t + \frac{\Delta{t}}{V_{t+{\Delta{t}}}}\frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}} \\
	    M'_{t+{\Delta{t}}} = \left( M'_t \cdot V_t \cdot \frac{1}{\Delta{t}} + \left( M'_t \cdot V_{t+{\Delta{t}}} \cdot \frac{1}{\Delta{t}} - M'_t \cdot V_{t+{\Delta{t}}} \cdot \frac{1}{\Delta{t}} \right) + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot V_t + \frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}} \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \\
	    M'_{t+{\Delta{t}}} = \left(  M'_t \cdot V_{t+{\Delta{t}}} \cdot \frac{1}{\Delta{t}}  + \left( M'_t \cdot V_t \cdot \frac{1}{\Delta{t}} - M'_t \cdot V_{t+{\Delta{t}}} \cdot \frac{1}{\Delta{t}} \right) + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot V_t + \frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}} \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \\
	    M'_{t+{\Delta{t}}} = \left(  M'_t \cdot V_{t+{\Delta{t}}} \cdot \frac{1}{\Delta{t}}  - \left( M'_t \cdot \frac{V_{t+{\Delta{t}}} - V_t}{\Delta{t}}  \right) + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot V_t + \frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}} \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \\
	    M'_{t+{\Delta{t}}} =  M'_t + \left(  - \left( M'_t \cdot \frac{V_{t+{\Delta{t}}} - V_t}{\Delta{t}}  \right) + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot V_t + \frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}} \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \\
	    M'_{t+{\Delta{t}}} =  M'_t + \left( \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot V_t + \frac{\Delta F_{t+{\Delta{t}}}}{\Delta{t}}  - \left( M'_t \cdot \frac{V_{t+{\Delta{t}}} - V_t}{\Delta{t}}  \right) \right) \cdot \frac{\Delta{t}}{V_{t+{\Delta{t}}}} \\
	    \text{short for:} \\
	    c(t_{i+1}) = c(t_i) + \left( GP(s) \cdot V + u_f - c(t_i) \cdot \frac{dV}{dt} \right) \cdot \frac{t_{i+1} - t_i}{V}
$$


在我们的实验中，我们有葡萄糖和谷氨酰胺进料，因此向量$\Delta F$​ 只有在对应于这两种物质的位置上有非零条目。



### 隐式建模补料

#### 训练

以某个浓度$i$的消耗为例:
$$
\frac{dM^i}{dt}= -k(\mathrm{M}^i) + \text{feed rate} \\
\frac{dM^i}{dt}= -k(\mathrm{M}^i) + \frac{\Delta F_{t+{\Delta t}}}{V_t \cdot \Delta t} \\
\text{where }\frac{dM^i}{dt} ≈ \frac{\Delta M^i_t}{\Delta t} = \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t} \\
y_{t}^i := \frac{M_{t+{\Delta t}}^i - M_{t}^i}{\Delta t} \cong  -k(\mathrm{M}^i) \cdot \frac{V_t}{V_{t+{\Delta t}}} + \frac{1}{V_{t+{\Delta t}}} \left( \frac{\Delta F_{t+{\Delta t}}}{\Delta t} - \left( M'_t \cdot \frac{V_{t+{\Delta{t}}} - V_t}{\Delta{t}}  \right) \right) \\
\text{short for:} \\
\frac{c(t_{i+1}) - c(t_i)}{t_{i+1} - t_i} \cong R(x_i) + \frac{1}{V} \left( u_f - c \cdot \frac{dV}{dt} \right) \equiv GP(x_i, u_f, V)
$$


#### 预测

$$
M'_0 = \text{InitialConcentration}(E') \\
M'_{t+{\Delta{t}}} = M'_t + \Phi((M'_t, \text{ExtraFeatures}(E'))) \cdot \Delta t \\
\text{short for:} \\
c(t_{i+1}) = c(t_i) + GP(x_i, u_f, V) \cdot (t_{i+1} - t_i)
$$

## 高斯过程回归

高斯过程回归 (GP) 是用于混合回归模型的一种特定选择。它可以近似一个函数 $f : \mathbb{R}^d \rightarrow \mathbb{R}$，其中 $d$​ 是特征的数量。

在这个简要概述中，为了简化，我们假设没有测量噪声，更多详细内容参见 (Rasmussen & Williams, 2006)。

### 先验定义

- 核函数: 
    - 高斯过程回归模型需要一个核函数或协方差函数 $k(\cdot, \cdot) : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$，这个核函数应该捕捉到适用于应用的相似性概念。
    - 使用核函数，定义了函数 $f(\cdot)$​ 的先验。

- 先验形式: 
    - 指定了在 $n+1$ 个点:
        - INPUT: $X \in \mathbb{R}^{(n+1) \times d}$ 
        - OUTPUT: 以及$X$ 对应的函数值 $Y \in \mathbb{R}^{n+1}$ 
    - 服从一个多元正态分布,  该分布, 
        - 均值为$\bold{0} \in \mathbb{R}^{(n+1)}$ 
        - 协方差矩阵 $K(X, X) \in \mathbb{R}^{(n+1) \times (n+1)}$ 
    - 通过将核函数应用于each pair row得到：

$$
Y \sim \mathcal{N}(0, K(X, X)), \\
K(X, X)_{ij} = k(x_i, x_j).
$$

### 预测分布

- 我们可以将这些数据点中的 $n$ 个视为观测数据（即训练数据 $X, Y$），其余的点作为仅已知 $x^*$ 的查询点。

- 通过使用观测训练数据**对先验进行条件化**，可以获得 $f(x^*) = y^*$​ 的预测分布 (Rasmussen & Williams, 2006, Section 2.2)：

$$
y^* | x^*, X, Y \sim \mathcal{N}\left(\bar{f}(x^*), Cov(\bar{f}(x^*))\right) \\
y^* = \bar{f}(x^*) = K(x^*, X)K(X, X)^{-1}Y \\
y_{cov}^* = Cov(\bar{f}(x^*)) = k(x^*, x^*) - K(x^*, X)K(X, X)^{-1}K(X, x^*) \\
$$



### 核函数

- **径向基函数核(RBF)**：
  
  - 平方指数核（Squared Exponential Kernel, SE）也称为高斯核或径向基函数（Radial Basis Function, RBF）。
  
  - 该核函数用于高斯过程回归中，具有以下形式：
    
    $$
    k(x, x') = \exp \left( -\frac{\|x - x'\|^2}{2\theta^2} \right)
    $$
    
  - 其中，$\|x - x'\|$表示输入向量 $x$ 和 $x'$ 之间的欧几里得距离，$\theta$ 是长度尺度超参数，控制了核函数对输入空间中距离变化的敏感性。
  
- **自动相关性确定（ARD）**：
  
  - 自动相关性确定（Automatic Relevance Determination, ARD）是对核函数的一种扩展，用于为不同的输入维度学习不同的长度尺度。
  - 对于每个输入维度 $i$，都有一个对应的长度尺度 $\theta_i，$核函数形式为：
    $$
    k(x, x') = \sigma \exp \left( -\sum_{i=1}^d \frac{(x_i - x_i')^2}{2\theta_i^2} \right)
    $$
  - 这种方法允许模型在不同维度上有不同的灵活性，从而能够更好地捕捉输入特征的相关性。
  - 本文使用的方法, 其中超参数:
  
      - $\bold{\theta} \in \mathbb{R}^d$ 是长度尺度（length scale）,  $d$​ 是特征的数量。
          - **定义**：长度尺度是核函数中用于衡量输入特征之间相似性的参数，通常位于 RBF 核函数的指数项中。
          - **作用**：它控制了输入特征空间中两个点之间的距离如何影响核函数的输出。较小的长度尺度表示模型对输入特征之间的小变化非常敏感，而较大的长度尺度表示模型对输入特征之间的变化较不敏感。
          - **优化**：长度尺度是核函数的关键参数，会在优化过程中调整，以找到最适合输入数据的距离尺度。
      - $\sigma \in R$ 是尺度因子（scaling factor）。
          - **定义**：尺度因子通常是核函数前的乘数，用于调整核函数的整体幅度。
          - **作用**：它控制了预测输出的整体幅度（方差）。更高的尺度因子意味着模型对输入数据的变化更加敏感，会产生更大的预测方差。
          - **优化**：尺度因子是核函数的一部分，会在优化过程中与其他参数一起调整，以找到最适合数据的整体幅度。
  
      - **不同的作用**：尺度因子控制核函数的整体幅度，而长度尺度控制输入特征之间的相似性。
      - **共同的目标**：在优化过程中，这两个参数共同作用，使得高斯过程回归模型能够最好地拟合训练数据并对新数据进行准确的预测。
      - 在高斯过程回归的优化过程中，尺度因子和长度尺度都是需要优化的参数。优化的目标是最大化训练数据在核函数下的似然，以找到最佳的参数组合。具体的优化过程通常使用最大似然估计（MLE）或贝叶斯优化等方法。



## 核函数的优化

### 单个时间

- $\theta$​ 的值可以通过最大化观测训练数据在先验下的似然来选择：

$$
\theta^* := \arg \max_{\theta} \log P(Y | X, \theta)
$$

### 多个时间

- 如何从$\mathbb{R}^d \rightarrow \mathbb{R}$  转变为$\mathbb{R}^d \rightarrow \mathbb{R}^t$
    - 通过使用共享超参数 $\theta$ 的 $t$ 个独立的单输出 GP。预测通过对每个目标使用 $t$ 次方程(16)来实现。

- 为了找到 $\theta$​​，可以通过最大化联合目标来实现：

$$
\theta^* := \arg \max_{\theta} \left( \sum_{t=1}^{t} \log P(Y^i | X, \theta) \right)
$$

- 对于单个时间下的单输出 GP 而言:

    - 训练样本的第 $i$​ 个目标的向量(例如 VCD, or GLC)

    - $Y^i \in \mathbb{R}^n$ 是 $n$ 个实验下的VCD 或其他目标的变化率 。
    - $X \in \mathbb{R}^{n \times d}$​ 

- 为了最大化联合目标, 可以将数据集进行 Observation-Wise Unfolding

    - $X \in \mathbb{R}^{(t*n) \times d}$​ 
    - $Y^i \in \mathbb{R}^{t*n}$ 



## 高斯过程模型集成方式

### 固定训练和验证集

- **训练与验证数据划分**：
  - 将全部数据集分为训练集和验证集，其中20%的数据用于验证，80%的数据用于训练。
- **具体过程**：
  - **数据划分**：将数据集分为训练集和验证集，其中20%的数据用于验证，80%的数据用于训练。
  - **模型训练**：训练10个不同的高斯过程模型，每个模型都使用相同的训练集。
  - **集成方法**：通过对10个模型的预测结果进行平均来获得最终的预测。



### 基于Bootstrap抽样随机训练和验证集

这种方法采用了简单的均值平均集成方法，通过对训练数据进行子采样来提高模型的鲁棒性和性能：

- **训练数据子采样**：
  - 从训练数据集中随机选择50% or 75%的数据进行训练。
  - 每次子采样都生成一个新的子集，从而形成多次子采样。
- **具体过程**：
  - 多次子采样训练数据，每次选择50% or 75%的数据。
  - 使用每个子采样的数据训练独立的高斯过程模型。
  - 对所有模型的预测结果取平均值，形成最终的预测结果。
- **优点**：
  - 高数量的模型和子采样确保了对训练数据的充分覆盖。
  - 提高了对过拟合的鲁棒性，因为每个模型只看到部分数据，从而降低了对单一数据集的依赖。
  - 集成模型通过平均多个模型的预测结果，能够减少单个模型的随机误差，提高总体预测性能。



## 不同产品的表示学习

在生物工艺应用中，我们希望训练一个高斯过程（GP）模型，该模型基于来自不同产品的过程数据进行训练, 这些产品表现出不同的行为。我们不使用产品的详细描述或特征化，而仅使用它们的名称。也就是说，对于每个数据点，我们只使用产品的身份。因此，我们需要向 GP 输入中添加一个分类特征，以表示数据点来自 $N_{\text{prods}}$ 个产品中的哪一个。然后，模型必须从训练数据中自行发现不同产品过程的相似性或差异性。这是否可行取决于产品身份的表示方式，以下将详细说明。

### One-Hot编码表示

传统上，分类特征通过One-Hot向量（类似于统计中的虚拟变量）表示。对于 $p \in \{1, \ldots, N_{\text{prods}}\}$ 的每个产品，定义一次性向量 $e_p \in \{0, 1\}^{N_{\text{prods}}}$，其在对应产品索引的位置上为 1。例如，第三个产品表示为向量 $e_3 = (0, 0, 1, 0, 0, 0)$。然后将一次性向量附加到其他特征上。因此，GP 的输入特征向量 $x = (f, e)$ 由过程状态特征 $f \in \mathbb{R}^d$（即当前浓度测量值、pH、温度等）和产品的一次性编码 $e \in \{0, 1\}^{N_{\text{prods}}}$ 组成。

为了更好地理解其影响，我们将公式中的和拆分开来，分别处理特征和产品表示：

$$
\text{RBF}_{\theta}(x, x') = \exp \left( - \sum_{i=1}^{d} \frac{(f_i - f_i')^2}{2\theta_i^2} - \sum_{j=1}^{N_{\text{prods}}} \frac{(e_j - e_j')^2}{2\theta_{d+j}^2} \right)
$$

可以拆分为：

$$
\text{RBF}_{\theta}(x, x') = \text{RBF}_{\theta}(f, f') \cdot \text{RBF}_{\theta}(e, e')
$$

其中 $f$ 和 $f'$ 是过程状态特征，$e$ 和 $e'$ 是产品的一次性编码。

因此，核函数只有在 $f$ 和 $f'$ 的核函数值较大且 $e$ 和 $e'$ 的核函数值较大的情况下才会较大。换句话说，只有当当前的浓度和过程条件相似，并且两个数据点来自相似产品时，$x$ 和 $x'$ 才被认为是相似的。

#### One-Hot编码的限制

虽然一次性编码可以有效地区分不同产品的数据，但它也存在一些限制：

1. **相似性信息的缺失**：
   - 一次性编码不携带不同产品之间相似性的信息。每个产品都是独立的，没有表达出任何可能的相似性或差异性。

2. **特征空间维度高**：
   - 对于大量的产品，编码向量的维度会非常高，增加了计算复杂度。

3. **泛化能力弱**：
   - 一次性编码无法捕捉到产品之间潜在的相似性或共性，因此在预测新产品时，模型的泛化能力可能会较弱。

通过一次性编码将产品身份添加到 GP 模型的输入中，可以使模型能够根据产品的身份进行预测。然而，由于一次性编码的限制，可能需要考虑其他更复杂的编码方法（如嵌入向量），以更好地捕捉产品之间的相似性和差异性，提高模型的预测性能和泛化能力。

#### 传统表示不能捕捉成对相似性

##### 假设与背景

假设我们有来自 $N_{\text{prods}} = 100$ 个不同产品的数据，这些产品形成了两个聚类：
- **聚类 A**：包括产品 1 和产品 2，它们彼此非常相似。
- **聚类 B**：包括产品 3 到产品 100，它们彼此非常相似，但与聚类 A 的产品不同。

我们希望使用高斯过程回归（GP）模型来捕捉这些产品之间的相似性。

##### One-hot编码的局限性

对于One-hot编码，每个产品用一个高维向量表示，该向量在产品对应的位置上为 1，其余位置为 0。

例如，产品 1 表示为 $e_1 = (1, 0, 0, \ldots, 0)$，产品 2 表示为 $e_2 = (0, 1, 0, \ldots, 0)$，依此类推。

在使用One-hot编码时，核函数 $\text{RBF}_{\theta}(e, e')$ 的形式为：
$$
\text{RBF}_{\gamma}(e, e') = 
\begin{cases} 
1 & \text{if } p = p' \\
\exp(-\gamma_p - \gamma_{p'}) & \text{if } p \neq p'
\end{cases}
$$

其中 $\gamma_j := \frac{1}{2\theta_{d+j}^2}$。

##### 思维实验

**聚类 A**：
- 产品 1 和产品 2 非常相似，理想情况下，它们之间的核相似性应该很高（接近 1）。
- 因此，我们期望 $\text{RBF}_{\gamma}(e_1, e_2)$ 应该是一个大值。

**聚类 B**：
- 产品 3 到产品 100 彼此非常相似，因此它们之间的核相似性应该很高。
- 因此，对于任何两个来自聚类 B 的产品 $i, j$，我们期望 $\text{RBF}_{\gamma}(e_i, e_j)$ 应该是一个大值。

**聚类 A 和聚类 B 之间的相似性**：
- 聚类 A 中的产品与聚类 B 中的产品应该有较低的相似性。
- 因此，对于聚类 A 中的任何产品 $p$ 和聚类 B 中的任何产品 $q$，我们期望 $\text{RBF}_{\gamma}(e_p, e_q)$ 应该是一个小值。

##### 分析与矛盾

根据公式 $\text{RBF}_{\gamma}(e, e')$：
- 如果 $p = p'$，则核函数为 1（完美相似性）。
- 如果 $p \neq p'$，则核函数为 $\exp(-\gamma_p - \gamma_{p'})$。

为了满足上述相似性要求，我们需要调整 $\gamma$ 参数：

1. **聚类 B 内部的高相似性**：
   - 对于聚类 B 内的产品（如产品 3 到 100），为了使 $\text{RBF}_{\gamma}(e_i, e_j)$ 是一个大值，我们需要 $\gamma_3, \ldots, \gamma_{100}$ 足够小。

2. **聚类 A 和聚类 B 之间的低相似性**：
   - 为了使聚类 A 和聚类 B 之间的相似性小，我们需要 $\gamma_1$ 和 $\gamma_2$ 足够大。

3. **聚类 A 内部的高相似性**：
   - 根据 $\text{RBF}_{\gamma}(e_1, e_2) = \exp(-\gamma_1 - \gamma_2)$，为了使产品 1 和 2 之间的相似性高，我们需要 $\gamma_1$ 和 $\gamma_2$ 足够小。

这就产生了矛盾：

- 为了让产品 1 和 2 之间的相似性高，我们需要 $\gamma_1$ 和 $\gamma_2$ 小。
- 但为了让聚类 A 和聚类 B 之间的相似性低，我们需要 $\gamma_1$ 和 $\gamma_2$ 大。

这种矛盾导致了传统one-hot性编码不能正确捕捉我们期望的相似性结构：

- **聚类 A 内部**：$\gamma_1$ 和 $\gamma_2$ 应该小，才能使 $\text{RBF}_{\gamma}(e_1, e_2)$ 大。
- **聚类 A 和 B 之间**：$\gamma_1$ 和 $\gamma_2$ 应该大，才能使 $\text{RBF}_{\gamma}(e_1, e_q)$ 和 $\text{RBF}_{\gamma}(e_2, e_q)$ 小（对于 $q \in [3, 100]$）。

由于不能同时满足这两种需求，传统的one-hot编码不能有效捕捉成对相似性，尤其是在复杂的聚类结构中。

一种更复杂的表示方法是使用**嵌入向量**，这些向量可以在高维空间中表示产品的相似性。这种方法可以捕捉产品之间的细微差异和相似性，而不是仅仅依赖于one-hot编码的有限表示能力。

通过使用嵌入向量，可以训练模型在高维空间中找到产品之间的相似性结构，从而提高模型的预测性能和泛化能力。嵌入向量可以通过学习在训练过程中优化，捕捉产品之间的细微相似性。



### 产品嵌入

#### 背景与问题
传统的一次性编码方法在捕捉产品相似性方面具有局限性，无法灵活地表示产品间的复杂相似性结构。因此，我们提出了一种新的方法，即使用产品嵌入向量来解决这个问题。

#### 产品嵌入向量的概念
- **嵌入向量**：每个产品 $ p \in \{1, \ldots, N_{\text{prods}}\} $ 由一个学习到的向量 $ w_p \in \mathbb{R}^D $ 表示，这些向量作为矩阵 $ W \in \mathbb{R}^{D \times N_{\text{prods}}} $ 的列。
- **输入特征**：在高斯过程（GP）的输入中，传统的高维一次性向量 $ e_p $ 被对应的低维嵌入向量 $ w_p $ 取代，该向量具有连续的条目。

#### 嵌入向量的优化
- **优化过程**：嵌入向量 $ W $ 在模型训练过程中通过优化确定，以最好地捕捉训练数据中观察到的产品相似性。
- **嵌入空间**：每个产品由一个点在抽象的 $ D $ 维嵌入空间中表示。

#### RBF 核函数的行为
- **核函数定义**：
  $$
  \text{RBF}_1(w, w') = \exp\left( -\frac{1}{2} \| w - w' \|^2 \right)
  $$
- **相似性判断**：两个产品被认为是相似的（高核值），当且仅当嵌入空间中的两个关联点在欧几里得距离上接近。
- **灵活性**：几乎任何产品的相似性结构都可以通过在足够高维的空间中选择适当的嵌入点来建模。

#### 嵌入向量的特性
- **不变性**：嵌入向量对旋转、平移和镜像不变。通过 $ \tilde{w}_i := A w_i + b $ 修改所有嵌入向量，其中 $ A \in \mathbb{R}^{D \times D} $ 是正交矩阵，$ b \in \mathbb{R}^D $，不会改变核值或 GP 的预测。
- **距离重要性**：只有嵌入点之间的距离重要，而每个坐标轴的确切值没有任何意义。

#### 实现步骤
1. **初始化嵌入向量**：为每个产品初始化一个嵌入向量 $ w_p $。
2. **优化嵌入向量**：在训练过程中，通过优化目标函数调整嵌入向量 $ W $ 的值，以最小化预测误差并捕捉产品之间的相似性。
3. **使用嵌入向量进行预测**：将产品的嵌入向量作为输入特征的一部分，进行模型预测。

通过使用嵌入向量代替一次性编码，可以更好地捕捉产品之间的相似性。这种方法灵活且具有更强的表达能力，适用于各种复杂的相似性结构，从而提高模型的预测性能和泛化能力。





### 优化产品嵌入向量

如前所述，嵌入向量在使用训练数据优化超参数时确定。具体来说，我们选择嵌入向量 $ W $ 和过程状态特征的长度尺度向量 $ \theta $，使其最大化如下的对数似然函数：

$$
\theta, W = \arg \max_{\theta', W'} \log P(Y | X, \theta', W')
$$

- $\theta$：只包含过程状态特征的长度尺度。
- $W$：包含产品嵌入向量。
- 在产品嵌入向量的维度上没有学习到的长度尺度，是固定的。

为了实现这一点，可以定义一个自定义核函数，其中包含可优化的超参数 $\theta$ 和 $W$：
$$
k_{\theta, W}(x, x') = \text{RBF}_{\theta}(f, f') \cdot \exp\left(-\frac{1}{2} \| W e - W e' \|^2\right)
$$

其中，$ e $ 和 $ e' $ 是产品的一次性表示。注意，$ W e $ 简单地从矩阵 $ W $ 中选取 $ e $ 中为 1 的列。

1. **定义自定义核函数**：
   - 自定义核函数结合了过程状态特征的 RBF 核和产品嵌入向量的相似性度量。

2. **优化过程**：
   - 通过最大化对数似然函数，优化自定义核函数的超参数 $\theta$ 和产品嵌入向量 $W$。

3. **实现方式**：
   - 将自定义核函数插入现有的 GP 超参数调整和预测算法实现中（例如，scikit-learn）。

通过引入产品嵌入向量并结合自定义核函数，可以更好地捕捉产品间的相似性。这种方法使模型在更高维空间中灵活表达产品之间的复杂关系，提高预测性能和泛化能力。





## 结果与讨论

湿实验（Wet-lab experiments）费用高且耗时，因此我们希望结合尽可能少的实验来开发新产品的工艺。

尤其是，我们希望在只有少量实验数据可用时，提高模型的准确性。

我们推测，包括来自其他产品的历史数据可以提高新产品预测的模型准确性。我们通过使用模拟数据的案例研究来调查这一点（见第2.5节）。

我们有5个历史产品（HP1到HP5），每个产品在相同的实验条件下进行了16次实验。历史产品的行为在图1中显示，展示了示例数据集在第一种实验条件下的结果。

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 15.02.35.png" alt="CleanShot 2024-05-31 at 15.02.35" style="zoom:50%;" />

此外，我们有一个新产品（NP），其数据仅来自少量实验条件（N_NP），这些条件不同于历史产品的16个实验条件。图2显示了示例数据集中新产品（N_NP=4）在不同实验条件下的过程数据。在每个为期14天的实验中，每天都进行测量,因此对于N_NP=4的情况，总共有4×14=56个数据点。

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 15.03.16.png" alt="CleanShot 2024-05-31 at 15.03.16" style="zoom:50%;" />

训练好的模型按照公式（10）和（12）用于预测完整的工艺发展（即每天的六个浓度测量值），并通过预测新产品NP的100个未见实验条件来评估模型性能。

### 选择嵌入维度

嵌入维度D的选择基于所有可用于训练的数据。它必须足够大，以便产品之间的适当成对距离可以在D维空间中实现。因此，我们选择对数边际似然(25)不再显著增加的D值，这表明最佳成对距离已经可以在该空间中实现。图3a显示了示例数据集（N_NP=4）的似然值。从中我们选择D=3，因为D=4的值不会进一步增加目标值。

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 15.07.37.png" alt="CleanShot 2024-05-31 at 15.07.37" style="zoom:50%;" />

为了验证这种选择嵌入维度的方法，我们使用统计方法进行了评估。我们生成了75对训练集（16×5+4次实验）和测试集（100次实验）。这些集的实验条件是独立地从拉丁方设计（Latin Hypercube Design）中抽取的，并且根据第2.5节模拟过程。因此，我们获得了75个独立的测试误差估计值。图3b显示了不同D值的测试误差分布的箱线图。我们观察到误差在D=3时减小，这证实了D=3是预期测试误差最低的选择。



### 产品嵌入提高了准确性

我们通过预测新产品 NP 的测试实验来评估所提出方法的性能。模型在来自5个历史产品的每个产品16次实验的数据和新产品的不同数量（N_NP）的实验数据上进行训练。比较了以下三种方法的性能：

1. 产品嵌入方法
2. 传统的一次性表示方法
3. 仅使用新产品 N_NP 可用实验次数训练的基线模型

测试集包含新产品 NP 上的100次实验，并重复比较150次，使用独立生成的训练集，以确保统计的有效性。



#### Titer表现

图4显示了测试误差分布的柱状图。结果表明：

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 16.20.58.png" alt="CleanShot 2024-05-31 at 16.20.58" style="zoom:50%;" />

- 当 N_NP = 4, 6, 8 时，使用历史数据的产品嵌入和一次性表示都比仅使用新产品的数据误差更低。
- **产品嵌入方法明显优于一次性表示方法**。例如，仅使用4次新产品实验的嵌入方法已经达到与使用8次实验的基线或使用6次实验的一次性表示相似的准确性。

**随着 N_NP 的增加（12, 16），使用历史数据的产品嵌入的好处减少**。如果有足够的新产品数据，可以直接建模其行为，无需扩展数据集到相关产品。



#### VCD表现

**然而，对于 N_NP = 16 的情况，基线模型在预测产量时优于嵌入算法**。这是因为**嵌入算法侧重于建模底层生物系统的动态**，而不是**产量**生产。嵌入算法倾向于将**具有相似细胞特征的产品分组，但这可能导致不同产量水平的错误预测**。

活细胞密度（VCD）预测的性能支持了这一假设（见图5）**对于 VCD，产品嵌入在所有 N_NP 值下都优于基线**，这表明**嵌入更有利于反映细胞特征**，**而不是产量形成**。

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 16.26.11.png" alt="CleanShot 2024-05-31 at 16.26.11" style="zoom:50%;" />

**通过为预测产量的任务设置单独的嵌入，可能会缓解这种情况**。



#### 结论

尽管如此，**对于产量和 VCD，我们都看到产品嵌入算法在数据较少时（即 N_NP = 4, 6, 8）的最大好处。**

通过案例研究的结果，**利用嵌入方法进行过程开发的历史数据优势变得显而易见**。这使我们能够在**仅进行少量实验后就使用准确的模型进行工艺决策**。



#### 应用方式

因此，从实际工作流程的角度来看，**对于新产品，可以从少量实验（校准实验——可能在模型跨产品显示最大差异的条件下进行）开始**，

然后使用嵌入方法分析和建模生成的数据。

随后，应利用嵌入模型设计下一组实验，以最大化感兴趣工艺条件周围的信息密度。

一旦为新产品生成了足够多的最大信息量的数据，可以根据分析的目的使用嵌入方法或仅使用 NP 方法进行分析。

模型在设计、优化或控制中的应用将决定哪种建模方法更适合当前应用，即平衡局部准确性与跨产品行为的一般描述。

无论如何，嵌入方法将有助于减少对新产品湿实验的需求，将实验放在重要的位置，从而加速工艺开发并降低成本。



### 可解释的嵌入向量

在第2.4.3节中，我们提出如果产品的嵌入点之间的欧氏距离较小，则这些产品是相似的。嵌入向量在[0.1, 3]^3的立方体中均匀随机初始化，并在训练过程中使用L-BFGS-B算法找到目标函数的局部最优解。每个产品都与抽象嵌入空间中的一个点（嵌入向量）相关联，这些点之间的距离反映了产品之间的相似性。

#### 嵌入向量距离分析

- 主要关注新产品 NP 与历史产品 HP1, ... HP5 之间的距离。

- 图6展示了训练前后嵌入空间中距离的分布。训练后的嵌入距离显著不同于随机初始化，特别是小的四分位间距表明嵌入可靠地捕捉了产品之间的相似性，并且对训练集不敏感。

    <img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 17.08.01.png" alt="CleanShot 2024-05-31 at 17.08.01" style="zoom:50%;" />

- 在150个训练集中，学习到的嵌入向量之间的距离变化很小。

- 具体来说，HP1 和 HP2 与 NP 最为相似，这并不意外，因为这三个产品都表现出乳酸消耗行为，而 HP3, HP4 和 HP5 则没有。

#### 嵌入向量的实际应用

<img src="Review-Datahow-2021-Simulation_Hybrid-Model_Discrete-Propagation-Model-GP.assets/CleanShot 2024-05-31 at 17.10.09.png" alt="CleanShot 2024-05-31 at 17.10.09" style="zoom:50%;" />

- 图7展示了从一个数据集学习到的产品嵌入向量。这些嵌入向量使用与图3a相同的示例训练集获得，展示了算法在实际数据集上的应用。
- 乳酸消耗产品 HP1 和 NP 围绕 HP2 形成一个群组，而产品 HP3 和 HP5 则松散地围绕 HP4 形成另一个群组。HP2 与新产品 NP 非常相似。
- 这种洞察可以帮助理解新产品的过程行为及其与已研究过的历史过程的关系，这是产品嵌入算法的一个重要应用。

- 图7所展示的定性结构在150次重复实验中大多数都得到了验证。
- 补充材料中包含类似图6的图表，展示了每个历史产品的距离分布，以确认这些观察结果的统计显著性。

总结：产品嵌入方法不仅提高了预测准确性，还通过捕捉产品间的相似性，提供了对新产品过程行为的深入理解，具有重要的实际应用价值。



## 限制与未来工作

在这项模拟研究中，我们故意保持设置的简单性，以便对高斯过程回归中的产品嵌入向量概念进行清晰简洁的介绍。因此，存在一些需要未来研究的限制。

- **测量噪声水平固定**：所有模拟的测量噪声水平都是固定的。随着噪声水平的增加，我们自然会预期所有回归模型的预测性能都会下降。然而，我们没有理由相信产品嵌入方法特别容易受到测量噪声的影响。尽管如此，未来的研究应对此进行深入探讨。

- **实验条件一致**：我们对所有五个历史产品使用相同的实验条件，因为在工业中实验设计通常在不同产品之间是一致的。然而，这只是一个简化假设，并非所提方法所必需。实际上，新产品的过程运行总是使用不同的实验条件。因此，我们认为，只要所有设计覆盖了相似的实验条件空间，即使使用不同设计，应该也能获得类似的结果。未来应详细研究设计选择的细微差别对方法性能的影响。

- **未与其他回归算法比较**：我们的案例研究未与其他回归算法进行比较。本文的目标是展示如何在高斯过程回归中特别有效地使用来自不同产品的历史数据。将产品嵌入方法适应于潜在的更优神经网络学习用于生物过程建模是很直接的。



## 结论

我们提出了一种用于多产品概率过程数据建模的嵌入方法，并通过上游哺乳动物细胞培养生物过程模拟案例研究对其进行评估。特别是，我们使用混合回归模型预测生物过程的演变。

- **预测性能提升**：通过使用所提出的嵌入方法，包括其他历史产品的数据，可以提高混合模型的预测性能。这使得模型能够从行为相似的过程数据中转移知识，从而减少训练准确模型所需的新过程数据。
- **嵌入空间可视化**：可视化嵌入空间可以为专家提供有关产品和过程之间复杂关系的宝贵见解。更重要的是，在预测过程中，它使算法能够对训练集中具有相似行为的产品的数据点给予更高的权重。因此，嵌入算法优于传统的一次性表示方法。
- **实际应用**：在只进行少量新过程实验后，利用现有历史数据，可以训练出高精度模型。例如，使用新产品的四次实验可以达到与使用八到十二次实验但不进行知识转移的情况类似的模型精度。
- **实验条件选择**：训练后的模型可以用于确定下一步应在湿实验中评估的过程条件。最简单的贪婪方案可以选择模型预测最高预期产量的过程条件。更复杂的方法可以利用高斯过程回归模型的不确定性来平衡开发和探索。

综上所述，我们相信所提出的算法将显著加快生物过程开发，或显著提高此类过程的质量预测能力。
