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





### 混合回归模型

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
  - $y_{t}^i$是表示浓度$i$的速率,用于建模中的标签

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



#### `预测`

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





