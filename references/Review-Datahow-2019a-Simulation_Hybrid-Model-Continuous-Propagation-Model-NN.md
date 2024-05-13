## **Abstract**

1. **挑战与解决方案：**
   
   - **挑战：** 建立适用于所有哺乳动物细胞培养过程的通用机理模型因缺乏对代谢网络和反应途径的完整理解而具有挑战性。
   - **解决方案：** 混合模型技术利用机理模型和数据驱动模型的协同作用，克服了单一方法的局限。

2. **研究发现与优势：**
   
   - 使用3.5升间歇供料实验的数据集进行分析，混合模型在预测不同过程变量的时间演变方面，相比传统统计模型展示出更高的准确性和外推能力。
   - 混合模型展现了优越的预测结果，包括更高的精度和稳健性，尤其在只利用初始和过程条件进行预测时更为明显。

3. **未来应用的重要性：**
   
   - 强调了混合模型在基于模型的过程优化和实验设计中的重要作用，尤其是在治疗蛋白生产领域。
   - 混合模型的应用不仅可以提高预测的准确性，还有助于定义更为稳健的过程设计空间，减少实验需求，加速产品开发流程。



## **Introduction**

**数据驱动模型（黑盒模型）：**

- **依赖性**：这类模型完全依赖于数据，通过输入和输出变量之间的统计相关性来捕捉相关的过程行为。
- **效率与简易性**：在算法性质简单且需求明确的情况下，这些模型可以高效且简单地生成，即使在专业知识有限的情况下也是如此。
- **数据需求**：需要大量高质量的数据来训练可靠的模型，但这些模型通常不适用于未被探索的区域。
- **功能**：可用于评估变量的重要性和相关性，识别过程噪声和异常的来源，并阐明非直观的相互关系，从而提出新的见解。
- **局限性**：这类模型不会生成新的过程知识。

**基于第一原理的模型（FPMs）：**

- **基础**：这些模型基于物理、化学和生物学原理，涵盖质量和能量平衡、热力学、传输现象和反应动力学方案。
- **表达形式**：通常以微分方程（常微分或偏微分）和代数方程的混合系统形式表达。
- **参数特性**：与数据驱动模型不同，FPMs中的参数具有明确的物理意义，因此通常可以在不依赖于具体过程的情况下预先估计。
- **可靠性与外推能力**：只要能捕捉到相关现象，这些模型就显示出高度的可靠性和良好的外推能力。
- **时间消耗与局限性**：模型的生成既耗时又复杂，当底层现象不被完全理解或缺少用于参数估计的变量测量时，创建FPMs变得不可能。

**总结**：

- **数据驱动模型**以其高效和简易的特点，在数据充足且问题定义明确时非常有效，但在未探索的区域和生成新过程知识方面存在局限。
- **基于第一原理的模型**则提供了一种基于深层物理、化学原理的可靠解决方案，适合在已知物理规律的情况下进行精确的问题处理，但其生成复杂且在某些情况下可行性受限。



**混合模型概念：**

- **定义：** 混合模型结合了机理模型框架和数据驱动方法，使用数据驱动方法来估计方程中的未知部分，并能灵活适应不同情境。
- **优势：** 机理结构的嵌入提高了模型的稳健性和外推能力，减少了过度拟合和所需数据量。
- **简化管理：** 数据驱动部分简化了系统复杂性的管理和模型参数及敏感度的估计。

**技术演进与应用历史：**

![](Review-Datahow-2019a-Simulation_Hybrid-Model-Continuous-Propagation-Model-NN.assets/748618c35348e43ac614f929dbe5d1c9a00f8259.png)

- **不同架构：** 
  - **串行方法(A, C)**：使用黑盒模型估计机理方程中的未知项。
  - **并行方法(B, D)**：使用黑盒模型减少机理模型的错误。
  - **混合方法**：使用机理模型生成数据来训练黑盒模型。

**应用范例：**

- **生物工程和化学工程的应用：**
  - 生物过程工程的例子包括普通生物反应器模型、青霉素生产过程、酵母和啤酒生产。
  - 化学工程的应用包括使用人工神经网络（ANNs）、支持向量回归、非线性偏最小二乘（PLS）等多种数据驱动模型。

**应用领域：**

- **模型预测控制(MPC)：** 混合模型已成功应用于模型预测控制，参考文献包括Sommeregger等人（2017年）和Von Stosch等人（2012年）的研究。
- **过程监测与预测(Process Monitoring)：** 在过程监测和预测方面，Von Stosch等人（2016年）和Zorzetto与Wilson（2003年）的工作体现了混合模型的有效性。
- **迭代过程优化(Iterative process optimization)：** 在迭代过程优化领域，Teixeira等人在2005年、2006年和2007年的系列研究中详细探讨了这一应用。
- **下游色谱过程：** Creasy等人（2015年）在治疗性蛋白制造的背景下，展示了混合模型在下游色谱过程中的应用。

**混合模型的构建与性能评估：**

- **模型构建：** 在本研究中，开发了一个基于人工神经网络（ANNs）和质量平衡方程的混合过程模型，用于预测单克隆抗体生产中间歇供料哺乳动物细胞培养生物反应器的关键状态变量的时间演变。
- **性能评估：** 所得混合模型的性能与最新的统计模型进行了比较，包括模型的准确性、插值和外推能力，以及在过程优化和实验设计中的潜在应用。



## **Dataset**

### **Data type**

- **数据来源**：本研究中开发的混合模型使用了最初由Rouiller等人（2012年）发布的细胞培养过程数据集进行测试。
- **实验设计**：数据集包括81个间歇供料runs，每次run的工作体积为3.5升，持续时间为10天。实验通过操纵三种种子条件（N-1扩增过程细胞密度、持续时间和细胞年龄）和两种过程条件（pH和DO设定点）进行。
- **参数变化**：
  - 接种密度: 扩增N-1过程细胞密度在5.51 × 10^6至7.17 × 10^6细胞/mL之间变化。
  - 细胞年龄: 在23至35天之间变化。
  - 扩增持续时间: 为4天或5天。
  - pH设定点: 在6.7到7.2之间变化
  - 溶解氧（DO）设定点: 在10%至70%之间变化。

- **Z矩阵**：表示所有设计条件的二维矩阵，行和列分别代表运行和操纵变量。
  - A_age
  - A_dur
  - A_Density
  - DO
  - pH
- **X矩阵**：表示动态变化的、非控制的过程变量（如可见细胞密度、葡萄糖浓度、乳酸、谷氨酰胺、谷氨酸和氨的浓度及渗透压），这是一个具有额外时间维度的三维矩阵。
  - Vcd
  - Glc
  - Lac
  - Gln
  - Glu
  - NH4
  - Osm
- **F矩阵**：组织在培养时间内添加到间歇供料中的不同代谢物的质量，这是一个与X矩阵维度相同的三维矩阵。
  - Glc
  - Gln
  - Glu
  - Lac
- **W矩阵**：建立的另一个过程信息矩阵，包括所有被控制以保持在特定设定点附近的变量，如pH、二氧化碳和氧的部分压力，这些变量每天测量。
  - pH,
  - pCO2
  - pO2
- **Y矩阵**：**产品滴度在0天至运行结束的交替日（即第2、4、6、8和10天）测量，对未量化的日子（即第1、3、5、7和9天）使用logistic interpolation进行估算**。
  - Titer
  
  - **logistic interpolation**
    
    逻辑插值（Logistic Interpolation）是一种在给定数据集上应用逻辑函数模型来估算未知数据点的方法。逻辑函数通常用于描述增长过程中的饱和现象，其形状类似于"S"形曲线。在Python中，可以使用`SciPy`库中的优化工具来实现逻辑插值。
    
    以下是使用Python进行逻辑插值的一个基本示例，我们将通过定义一个逻辑函数并使用`curve_fit`方法从`scipy.optimize`来拟合给定的数据点。
    
    ### 步骤 1: 定义逻辑函数
    
    逻辑函数通常定义为：
    
    $f(x) = \frac{L}{1 + e^{-k(x-x_0)}}$
    
    其中：
    - \( L \) 是曲线的最大值，
    - \( k \) 是曲线的陡峭程度，
    - \( x_0 \) 是曲线的中点。
    
    ### 步骤 2: 准备数据
    
    假设你有一组实验数据，你需要在这组数据上应用逻辑插值。
    
    ### 步骤 3: 使用 curve_fit 进行参数拟合
    
    利用`curve_fit`从`scipy.optimize`可以很方便地找到逻辑函数的最佳参数。
    
    ### 示例代码
    
    ```python
    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    
    # 定义逻辑函数
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))
    
    # 示例数据 (x值和y值)
    xdata = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ydata = np.array([0.1, 0.15, 0.3, 0.6, 1.1, 2, 3.2, 4.5, 5.8, 6.0])
    
    # 使用curve_fit进行逻辑函数拟合
    params, covariance = curve_fit(logistic, xdata, ydata, p0=[max(ydata), 1, np.median(xdata)])
    
    # 打印最优参数
    print("L =", params[0])
    print("k =", params[1])
    print("x0 =", params[2])
    
    # 绘制数据点
    plt.scatter(xdata, ydata, color='red', label='Data Points')
    
    # 绘制拟合曲线
    xmodel = np.linspace(min(xdata), max(xdata), 300)
    ymodel = logistic(xmodel, *params)
    plt.plot(xmodel, ymodel, label='Fitted Curve')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Logistic Curve Fit')
    plt.legend()
    plt.show()
    ```
    
    在这个示例中：
    
    - 我们定义了一个逻辑函数`logistic`。
    - 准备了一些模拟数据`xdata`和`ydata`。
    - 使用`curve_fit`进行了参数拟合，并得到了最佳拟合参数。
    - 使用`matplotlib`展示了数据点和拟合曲线。
    
    逻辑插值非常适合于那些数据随着时间（或其他因素）逐渐接近某个极限值的情况，常见于生物学和化学过程建模等领域。



### **Data type function**

- **矩阵Z, X, W的功能**：
  
  ![](Review-Datahow-2019a-Simulation_Hybrid-Model-Continuous-Propagation-Model-NN.assets/805a1f2fb7c1dcb87ebed80ce086f963bf45fa67.png)
  
  - **Z矩阵**：在整个模型运行期间保持不变，表示固定的实验条件或参数。
  - **X矩阵**：动态信息，反映了随时间变化的过程变量，如细胞密度或化学物质浓度，这些在黑盒模型中被机理框架计算。
  - **W矩阵**：也包含动态信息，通常涉及可直接测量的实验参数，如pH和氧的部分压力等。

- **模拟和计算过程**：
  
  - 每天培养过程中相关的质量平衡是如何整合的，以及如何模拟添加饲料来为下一天的过程做准备。
    - **新起点的设置**：
      每次添加饲料后，原有的质量平衡会发生变化，需要重新计算和设定新的起点，以便于下一天的培养可以在更新后的条件下继续进行。
  - Y矩阵，尽管包含了滴度信息，但在模型计算中并未使用，因为滴度由模型直接计算，故Y矩阵扮演了数据监控而非计算的角色。
- **数据插补**：
  - 文中提到大约4%的数据缺失，并使用了削减分数回归(Trimmed Score Regression)算法进行数据插补，这是一种用于处理缺失数据的统计技术，以确保数据完整性，使模型预测更为准确。

- Trimmed Score Regression: 是一种处理异常值和偏差数据的回归技术，它通过修剪（削减）部分数据点的贡献来提高统计估计的鲁棒性。此方法特别适用于数据集中存在异常值或非典型观测值时，有助于得到更为稳健的回归结果。

    实现Trimmed Score Regression通常涉及以下几个步骤：

1. **拟合初始模型**：
   首先对数据拟合一个标准的回归模型（如线性回归），以获得初步的参数估计。

2. **计算残差**：
   基于初步模型计算每个数据点的残差（实际观测值与模型预测值之间的差异）。

3. **残差排序**：
   将所有数据点的残差按绝对值大小进行排序。

4. **数据削减**：
   移除具有最大残差的一定比例的数据点，这些通常被认为是潜在的异常值或极端观测值。

5. **重新拟合模型**：
   使用剩余的数据点重新拟合回归模型。

6. **迭代优化**（可选）：
   这个过程可以迭代进行，每次迭代中进一步调整削减的数据点，直到满足停止准则（如参数估计的变化非常小）。

下面是一个使用Python进行Trimmed Score Regression的简单示例。我们将使用`statsmodels`库来拟合线性回归，并手动实现数据削减的步骤：

```python
import numpy as np
import statsmodels.api as sm

# 生成一些测试数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = 3 * X + np.random.normal(0, 2, 100) + 20 * np.random.binomial(1, 0.05, 100)  # 添加一些异常值

# 添加常数项
X = sm.add_constant(X)

# 拟合初始线性回归模型
model = sm.OLS(y, X).fit()

# 计算残差
residuals = np.abs(model.resid)

# 排序残差并找到削减点（例如削减最大的5%的数据点）
cutoff = np.percentile(residuals, 95)

# 保留残差小于削减点的数据
X_trimmed = X[residuals < cutoff]
y_trimmed = y[residuals < cutoff]

# 使用削减后的数据重新拟合模型
trimmed_model = sm.OLS(y_trimmed, X_trimmed).fit()

# 打印两个模型的参数进行比较
print("Original model params:", model.params)
print("Trimmed model params:", trimmed_model.params)
```

- 这种削减方法会改变数据的分布，因此在进行削减之前应当仔细考虑削减比例。
- 削减分数回归是一种非参数方法，适用于**有较强异常值的数据集**。
- 实际应用中可能需要根据具体情况调整削减的策略和比例。



### **Data Split and Usuage Methodology**

1. **数据划分**：数据集被划分为校准集（训练集）和测试集，比例通常是80%用于训练，20%用于测试。这种划分是为了在一部分数据上训练模型，而另一部分数据用来评估模型的泛化能力。

2. **模型训练**：使用校准集来训练模型。在这个阶段，可能会使用多种不同的模型或算法。

3. **超参数调整**：通过交叉验证来调整模型的超参数。交叉验证是一种统计学方法，用来评估并改善模型对独立数据集的预测能力，它通过多次分割数据集并重复训练模型来工作。

4. **性能评估**：使用根均方误差（RMSEP）作为性能指标来评估不同模型的预测精度。RMSEP是实际观测值与模型预测值之间差的平方的平均值的平方根，它提供了误差的量度，较低的RMSEP值通常意味着更好的模型性能。

以下是使用Python进行上述过程的示例代码，包括数据划分、模型训练、交叉验证和性能评估：

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# 生成一些回归数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置不同的模型配置
models = {
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params': {
            'alpha': [0.1, 1.0]
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf']
        }
    }
}

# 使用GridSearchCV进行交叉验证和超参数调整
for model_name, model_setup in models.items():
    grid_search = GridSearchCV(model_setup['model'], model_setup['params'], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    rmsep = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} RMSEP: {rmsep:.4f}")

```

这段代码使用了随机森林回归模型，它是一种广泛使用的强大的机器学习模型，适用于回归和分类问题。在实际应用中，可能需要根据具体问题调整模型类型和参数。



### **Hybrid Model Design**

当前工作采用了一种串行架构的混合建模方法（Thompson & Kramer, 1994），如图1b的示意流程图所示。代表细胞培养的方程体系是基于质量平衡建立的，如下所报告：  
dCi/dt = μi(t)C1(t)，其中T+ ≤ t < T+1,  
其中T从培养第0天变化到第9天，T+表示喂料后的时间点，Ci和μi分别是第i种物种的浓度和特定速率，i代表Xv（可行细胞密度）、GLC（葡萄糖）、LAC（乳酸）、GLN（谷氨酰胺）、GLU（谷氨酸）、NH4（氨）、Osm（渗透压）和滴度。因此，有八个建模目标，C1代表Xv。由于对细胞代谢的理解不完全，特定速率常常事先未知。这种知识的缺乏通过图1b中示意的黑箱模型得到补偿，该模型基于培养实验的信息估计特定速率。  
μ(t) = f(Xv, GLC(t), LAC(t), GLN(t), GLU(t), OSM(t), Z, W(t), NH4(t)),  
任何符合方程(2)所述形式的回归工具都可以用来将过程因子和过程变量测量映射到μs。特别是，本研究中使用了一种前馈单隐层人工神经网络，它自动考虑了由于其非线性结构而导致的输入的不同转换。ANN公式和激活函数的详细解释可以在（Von Stosch等人，2016年）的文献中找到。本质上，积分和优化是同时进行的，以优化神经网络权重，使测量值和模型预测浓度值之间的差异最小化。积分是针对一天的时间跨度进行的，最终，喂料模拟如下：
Ci,init(T) = Ci(T) + Feedi(T)/V,  
其中Ci(T)代表喂料前的浓度值，也对应于测量值，而Ci,init(T)代表从T到T+1积分方程(1)的初始条件，Feedi(T)是喂入的物种i的质量，V是反应器容积。使用二范数正则化目标函数（Yang等人，2011）来避免神经网络权重的过拟合，并使用五折交叉验证（Hastie, Tibsharani, & Friedman, 2009）来确定最优节点数和正则化参数。

- **模型建立**：首先建立了一个基于质量平衡的微分方程系统，这些方程描述了培养过程中各种化学物质的浓度变化。
- **黑箱模型补偿**：由于缺乏对细胞代谢完整的理解，模型中的某些参数（如化学物质的变化速率）可能无法直接测量或预测。为此，使用黑箱模型（如人工神经网络）来估计这些参数。
- **神经网络优化**：

使用人工神经网络来处理非线性关系，并优化网络权重，以减少实际测量值和模型预测值之间的差异。

- **正则化和交叉验证**：为了避免过拟合，使用正则化技术，并通过交叉验证来选择最优的模型配置，如网络的层数和节点数量。

下面是一个简化的Python代码示例，展示如何使用`scikit-learn`库训练一个具有正则化的简单神经网络来拟合数据：

```python
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 生成模拟数据
X, y = make_regression(n_samples=200, n_features=10, noise=0.1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练神经网络模型
model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', alpha=0.001, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 预测和评估
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")
```

在这个示例中，我们使用了`MLPRegressor`来创建一个单隐层的前馈神经网络，并通过调整正则化参数`alpha`来控制模型的复杂度，从而防止过拟合。此外，我们还使用了交叉验证来评估模型的性能。
