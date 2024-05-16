## Abstract & Introduction

Hybrid-EKF该框架结合了混合模型和扩展卡尔曼滤波器（EKF），用于哺乳动物细胞培养过程的实时监控、控制和自动决策制定。该研究的主要贡献包括：

1. **改进预测监控准确性**：通过与传统的基于PLS模型的工业标准工具相比较，使用Hybrid-EKF框架能在特定应用中将预测监控准确性提高至少35%。

2. **工业应用优势**：研究强调了Hybrid-EKF在工业应用中的优势，尤其是在条件性过程喂养和过程监控方面。例如，将Hybrid-EKF作为产量（titer）的软传感器使用时，与最先进的软传感器工具相比，预测准确性提高了50%。

3. **提供更有效的过程监控和控制方法**：通过Hybrid-EKF框架，实现了对哺乳动物细胞培养过程的更有效监控和控制，支持了行业4.0中智能工厂的构建，其中的工厂能够适应不同的过程场景并自主运作。



## Data organization

<img src="Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.07.42@2x.png" alt="CleanShot 2024-05-15 at 16.07.42@2x" style="zoom:50%;" />

<img src="Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.09.25@2x.png" alt="CleanShot 2024-05-15 at 16.09.25@2x" style="zoom:50%;" />



### 实验设计因子:

- Xv_0
- Glc_0
- Gln_0
- pH_before
- pH_after
- pH_shift_day
- Temp_before
- Temp_after
- Temp_shift_day
- Feed_Gln
- Feed_Glc
- Feed_start
- Feed_end



### In-silico Dataset

这个数据集使用宏观动力学模型（引用了Craven等人和Xing等人的研究）进行模拟，这些模型针对哺乳动物细胞培养进行了调整，以体现生长率依赖性的复杂非线性以及对pH和温度变化的考量。主要要点如下：

<img src="Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.26.45.png" alt="CleanShot 2024-05-15 at 16.26.45" style="zoom:50%;" />

1. **数据模拟与设计**：数据集包含14个过程因子，例如初始的细胞密度（Xv）、葡萄糖（GLC）、谷氨酰胺（GLN）浓度、pH值在变化前后的情况、温度在变化前后的情况、变化的天数、葡萄糖和谷氨酰胺的添加开始与结束日期，以及这些添加物的日均量。使用分数因子设计进行了100次实验的模拟，每次实验持续14天。

2. **数据结构与组织**：模拟的过程数据按不同信息源组织，包括动态变化的非控制过程变量如Xv和GLC，这些以三维矩阵X表示，时间是第三维。此外，实验中控制的变量如pH和温度以W矩阵形式表示，而产品特性由Y矩阵表示。
    - X
        - Xv
        - GLC
        - LAC
        - GLN
        - NH4
        - Titer
    - W:
        - pH
        - temp
    - F:
        - Feed_GLC
        - Feed_GLN
    - Z: none
    - 

3. **数据测量与噪声**：模拟数据中的测量频率为2.4小时一次，并加入了**15%的高斯噪声**，以模拟光谱技术的测量误差。**pH和温度的动态变化被纳入考量**，并且**每天都有葡萄糖和谷氨酰胺的添加**，这些都是设计中的因素之一。

4. **模拟数据的应用**：这些模拟数据被用来测试Hybrid-EKF框架，评估混合模型在实时监控和控制哺乳动物细胞培养中的性能，并与现实工业数据集进行比较。

5. 虽然使用了 100 个实验进行训练,但是实际上 16 个实验即可达到稳健的性能,如下图

<img src="Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.28.52@2x.png" alt="CleanShot 2024-05-15 at 16.28.52@2x" style="zoom:50%;" />



### Real Dataset

数据集包含81次3.5升工作体积的连续进料批次实验，每次实验持续10天。

- X
    - Xv
    - GLC
    - LAC
    - GLN
    - GLU
    - NH4
    - OSM
    - measured once per day
- W:
    - pH
    - pO2
    - pCO2
    - measured once per day
- F:
    - Feed_GLC
    - Feed_GLN
- Z:
    - pH_set
    - DO_set
- Y: Titer
    - measured on even days from day 0 until the end of the run



### 数据集分割

<img src="Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.49.19.png" alt="CleanShot 2024-05-15 at 16.49.19" style="zoom:50%;" />

### Evaluation Metric

Root Mean Squared Error in Prediction (RMSEP)



## Model

### Hybrid model

![CleanShot 2024-05-15 at 16.53.11](Review-Datahow-2020-Process-Monitoring_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 16.53.11.png)

#### Input setting:

- X:

    - In-silico dataset
        - i for : Xv, GLC, GLN, LAC, NH4, titer

    - Real dataset
        - i for : Xv, GLC, GLN, LAC, NH4, titer, GLU Osm

- W

- Z

#### ANN setting:

- Single Layer:
    - In-silico dataset: 8
    - real dataset: 10
- Training:
    - Five-fold
    - L2 regularization
    - 



 



### Historical model

#### Historical-PLS2

PLS2模型（偏最小二乘回归）是一种多变量统计技术，用于建立输入变量和输出变量之间的关系。在该模型中，为每个时间点构建一个PLS2模型，以映射直到该时间点的所有可用数据到下一时间点的状态。因此，Historical-PLS2模型在每个时间点都有一个独立的模型。

具体来说，该模型的映射关系表示如下：
$$
[Z, X(0 < t \leq t_{\text{model}}), W(0 < t \leq t_{\text{model}})] \xrightarrow{\text{PLS2}} \text{State}(t_{\text{model}}+1)
$$

其中，$ t_{\text{model}} $表示PLS2模型开发的时间点，States是指X和Y变量。在预测阶段，对于PLS传播，只有在 $ t = 0 $ 时的输入（即过程设计, Z, W0, X0）被提供给Historical-PLS2模型。然后，模型预测时间 $ t = 1 $ 的值，并将其作为输入用于预测下一个时间点 $ t = 2 $，依此类推。

对于数据集 `Z`, `X`, `W`, `Y` 

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression

class HistoricalPLS2:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.models = []

    def fit(self, Z, X, W, Y):
        T = X.shape[1]  # Number of time points
        for t in range(1, T):
            # Collect data up to time t
            X_t = X[:, :t, :].reshape(X.shape[0], -1)
            W_t = W[:, :t, :].reshape(W.shape[0], -1)
            inputs = np.hstack([Z, X_t, W_t])
            outputs = np.hstack([X[:, t, :], Y[:, t, :]])
            model = PLSRegression(n_components=self.n_components)
            model.fit(inputs, outputs)
            self.models.append(model)

    def predict(self, Z, X_0, W_0):
        T = len(self.models)
        X_pred = [X_0]
        W_pred = [W_0]
        for t in range(T):
            model = self.models[t]
            X_t = np.array(X_pred).reshape(1, -1)
            W_t = np.array(W_pred).reshape(1, -1)
            inputs = np.hstack([Z, X_t, W_t])
            output = model.predict(inputs)
            X_next = output[:, :-1]
            X_pred.append(X_next)
            W_pred.append(W_0)
        return np.array(X_pred).reshape(-1, X_0.shape[-1])

# 示例数据（需根据实际数据替换）
Z = np.random.rand(100, 5)  # 100 runs, 5 operating conditions
X = np.random.rand(100, 10, 6)  # 100 runs, 10 time points, 6 variables
W = np.random.rand(100, 10, 3)  # 100 runs, 10 time points, 3 control variables
Y = np.random.rand(100, 10, 1)  # 100 runs, 10 time points, 1 target variable

# 拟合模型
historical_pls2 = HistoricalPLS2(n_components=2)
historical_pls2.fit(Z, X, W, Y)

# 预测
Z_new = np.random.rand(1, 5)
X_0_new = np.random.rand(1, 6)
W_0_new = np.random.rand(1, 3)
X_pred = historical_pls2.predict(Z_new, X_0_new, W_0_new)

print(X_pred)

```

1. `HistoricalPLS2` 类构建并管理多个PLS2模型。
2. `fit` 方法用于训练每个时间点的PLS2模型。
3. `predict` 方法用于逐时间点预测状态。



#### PLS-direct

PLS-direct 是一种改进的 PLS2 模型方法，不同于 Historical-PLS2。PLS-direct 使用实际测量值而不是前一个模型的预测值或校正后的状态来进行下一步预测。这意味着在每个时间点 $ t $，模型使用时间点 $ t $ 的实际测量值来预测时间点 $ t+1 $ 的状态。

- Historical-PLS2 和 PLS-direct 的区别

    - **Historical-PLS2**:

        - 这种方法在每个时间点 $ t $ **使用之前所有时间点的数据**来预测下一个时间点 $ t+1 $ 的状态。

        - 在**预测阶段**，**使用前一个时间点的预测值作为输入进行下一时间点的预测**。这意味着**误差可能会累积**。

    - **PLS-direct**：
        - 与 Historical-PLS2 相似，**但在每个时间点 $ t $ 使用实际测量值而不是前一个时间点的预测值来进行下一步预测**。
            - 这样可以**避免误差累积**，因为它依赖于实际测量值而不是逐步预测。



#### PLS 最优 n-components

在现有的 `HistoricalPLS2` 类基础上，实现五折交叉验证来确定每个时间点最优的潜在变量数目。可以使用 `GridSearchCV` 进行交叉验证。以下是更新后的代码：

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

class HistoricalPLS2:
    def __init__(self, max_components=10):
        self.max_components = max_components
        self.models = []

    def fit(self, Z, X, W, Y):
        T = X.shape[1]  # Number of time points
        for t in range(1, T):
            # Collect data up to time t
            X_t = X[:, :t, :].reshape(X.shape[0], -1)
            W_t = W[:, :t, :].reshape(W.shape[0], -1)
            inputs = np.hstack([Z, X_t, W_t])
            outputs = np.hstack([X[:, t, :], Y[:, t, :]])

            # Define PLS regression model and parameters for grid search
            pls = PLSRegression()
            param_grid = {'n_components': range(1, self.max_components + 1)}
            grid_search = GridSearchCV(pls, param_grid, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
            grid_search.fit(inputs, outputs)
            
            # Select the best model
            best_model = grid_search.best_estimator_
            self.models.append(best_model)

    def predict(self, Z, X_0, W_0):
        T = len(self.models)
        X_pred = [X_0]
        W_pred = [W_0]
        for t in range(T):
            model = self.models[t]
            X_t = np.array(X_pred).reshape(1, -1)
            W_t = np.array(W_pred).reshape(1, -1)
            inputs = np.hstack([Z, X_t, W_t])
            output = model.predict(inputs)
            X_next = output[:, :-1]
            X_pred.append(X_next)
            W_pred.append(W_0)
        return np.array(X_pred).reshape(-1, X_0.shape[-1])

# 示例数据（需根据实际数据替换）
Z = np.random.rand(100, 5)  # 100 runs, 5 operating conditions
X = np.random.rand(100, 10, 6)  # 100 runs, 10 time points, 6 variables
W = np.random.rand(100, 10, 3)  # 100 runs, 10 time points, 3 control variables
Y = np.random.rand(100, 10, 1)  # 100 runs, 10 time points, 1 target variable

# 拟合模型
historical_pls2 = HistoricalPLS2(max_components=10)
historical_pls2.fit(Z, X, W, Y)

# 预测
Z_new = np.random.rand(1, 5)
X_0_new = np.random.rand(1, 6)
W_0_new = np.random.rand(1, 3)
X_pred = historical_pls2.predict(Z_new, X_0_new, W_0_new)

print(X_pred)
```

1. **初始化 `HistoricalPLS2` 类**：增加 `max_components` 参数，指定最大潜在变量数目。
2. **fit 方法**：
   - 在每个时间点收集数据并进行拼接。
   - 定义 PLS 回归模型，并使用 `GridSearchCV` 进行五折交叉验证，确定最优的潜在变量数目。
   - 选择最优模型并保存到 `self.models` 列表中。
3. **predict 方法**：保持与之前相同的逻辑，使用训练好的模型逐时间点进行预测。

