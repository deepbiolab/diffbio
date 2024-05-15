## 摘要

- **混合半参数模型：** 结合了机制和机器学习方法，用于生物过程的开发。
- **引导聚集（Bootstrap Aggregation）：** 提出用于增强模型预测能力，特别是当使用统计设计实验数据时。
- **实验设计方法：** 包括Box-Behnken、中心复合和Doehlert设计，旨在识别最佳的细胞生长和重组蛋白表达条件。
- **预测性能的改善：** 引导聚集显著减少了对所有三种设计的新批次实验的预测均方误差。
- **模型校准：** 聚集的**最优模型数量**是关键参数，需要针对每个具体问题进行调整。
- **Doehlert设计的优势：** 在识别过程最优条件方面略优于其他设计方法。
- **误差界限的计算：** 多个预测的可用性允许对模型各部分的误差界限进行计算，提供了对模型组件预测变异的深入了解。

## Introduction

- **混合模型的定义**：混合半参数模型结合了机器学习方法（非参数化，如人工神经网络、偏最小二乘法、支持向量机）和质量守恒定律（参数化），用于从过程数据中学习未知或不完全理解的细胞动力学/动态。
- **实际优势**：这种模型可以减少过程开发中所需的实验数量，提高新（最优）过程条件的预测能力。
- **数据的重要性**：建立非参数组成部分的数据量、质量和结构至关重要。数据通常被分为训练集、验证集和测试集，以支持模型的参数估计、验证和独立数据性能评估。
- **数据分割方法**：数据分割是一种抽样问题，可以通过重新抽样方法来解决。例如，bagging（引导聚集）是一种基于重新抽样的集成方法，已成功应用于神经网络、偏最小二乘模型和决策树。
- **集成方法在混合模型中的应用**：集成方法如bagging和boosting在混合模型中的应用还相对有限，**但这些方法有助于评估从混合模型的一个部分到另一个部分的不确定性传播(propagation of uncertainty)**，这对于过程控制和优化尤其重要。



## 方法

### 案例研究：大肠杆菌连续投料过程

本研究选用了一个先前建模并优化过的大肠杆菌连续投料过程作为案例研究。通过过程模拟生成了合成数据集，这样可以公正地比较不同的建模方法，避免未知的生物和/或实验变异带来的偏差。

模拟模型的详细信息在“附录A”中提供。简要来说，该模型通过应用质量守恒定律，描述了在搅拌罐连续投料生物反应器中生物量、底物和产物浓度的动态变化。具体的生长、底物吸收和产物形成的动力学速率被定义为底物浓度和温度的非线性函数，使用Monod型动力学，其中温度依赖性决定了最大可实现的速率。设计因素包括温度$T$（变化范围为 29.5 至 33.5°C）、特定生长速率设定点 $\mu_{\text{Set}}$（变化范围为 0.1 至 0.16 h^(-1)）和诱导时的生物量浓度$X_{\text{ind}}$（变化范围为 5.0 至 19.0 g/kg）。模型针对不同设计因素的值进行了模拟，应用了三种统计学上不同的设计方法：

<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/image-20240514224228786.png" alt="image-20240514224228786" style="zoom:50%;" />

1. **内嵌中心复合设计 (CCD)**：进行了17次培养实验（包括中心点的两次重复）和249个测量点；
2. **Box-Behnken设计 (BBD)**：进行了15次培养实验和222个测量点；
3. **Doehlert设计 (DD)**：进行了15次培养实验和214个测量点。

**设计因素的范围**:

- **特定生长速率设定点 (μ_set)**：从0.1到0.16 1/h，影响生物量的增长速度。
- **温度 (T)**：不同设计中温度范围略有不同，总体在大约28.5°C到33.5°C之间变化，影响生物过程的活性和效率。
- **诱导时的生物量浓度 (X_ind)**：各种设计中从5.0到19.0 g/kg不等，这是开始生产目标蛋白质或其他代谢产物前的生物量浓度。



<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/image-20240514225152662.png" alt="image-20240514225152662" style="zoom:50%;" />

这个表格展示了使用Box-Behnken设计（BBD）进行的一个大肠杆菌连续投料实验过程的详细数据记录。表中记录了实验过程中不同时间点的各种参数和过程动态。

表格列说明：

1. **Exp**：实验编号，表明这些数据属于同一个实验批次。
2. **Time (h)**：时间，以小时为单位，展示了实验从开始到结束的每个测量时间点。
3. **muSet (1/h)**：特定生长速率设定点，是生物反应过程中控制生物生长速率的参数。
4. **TempSet (°C)**：设置的温度，用于控制生物反应器内的温度。
5. **Xind (g/kg)**：诱导时的生物量浓度，表示在诱导表达开始时的生物量浓度。
6. **Biomass (g/Kg)**：不同时间点测量的生物量。
7. **Substrate (g/kg)**：底物浓度，显示生物反应过程中底物的消耗情况。
8. **Product (U/kg)**：产物浓度，表示不同时间点的产物（如蛋白等）浓度。
9. **CultureMass (kg)**：培养质量，反应器内的总质量。
10. **uF (kg/h)**：上流量，表示每小时向反应器中添加的流体量。
11. **Temp (°C)**：实际测量的温度，显示实验过程中的温度变化。
12. **Induction (on/off)**：表达诱导的开/关状态，1 表示诱导打开，0 表示关闭。

数据分析：

- 随着时间的推移，生物量、产物浓度逐渐增加，显示了生物反应的进行和产物的积累。
- 诱导表达（Induction）在第一次测量后（约4.7小时）被打开，随后产物浓度显著增加，从37.66 U/kg跃升至最后的434.07 U/kg。
- 生物量从开始的4.00 g/kg增加到最后的33.07 g/kg，说明生物质的积累。
- 底物浓度在诱导表达前保持较低（0.37 g/kg以下），诱导后略有增加，这可能与生物代谢活动增强有关。



### 模型描述

1. **基础公式**:
   - **物质守恒定律**：用方程 $ \frac{dc}{dt} = K \cdot r(c, x) - D \cdot c + u $ 描述，其中 $ c $ 是浓度向量，$ K $ 是已知的产率系数矩阵，$ D $ 是稀释率，$ u $ 是体积进料率（控制输入），$ r(c, x) $ 是体积反应速率向量。
   
2. **反应速率函数**:
   - **$ r(c, x) $**：是一个复杂的非线性函数，依赖于浓度 $ c $ 和其他物理化学属性 $ x $（例如温度和 pH）。该函数的具体形式为 $ r(c, x) = r(g(f(c, x), w), c, x) $，其中 $ r(g, c, x) $ 是基于已知知识的参数化函数，而 $ g = g(f(c, x), w) $ 是代表需要从数据中"学习"的未知现象的非参数化函数。
   - **神经网络模型**：$ g $ 函数通常通过一个简单的前馈神经网络实现，该网络包含三层，转换函数在输入和输出层是线性的，隐藏层是双曲正切函数。神经网络的参数向量 $ w $ 需要通过数据来识别。

3. **材料平衡方程**：
   - **生物量和产品**：材料平衡方程用于生物量 $ X $ 和产品 $ P $，形式为 $ \frac{dX}{dt} = \mu \cdot X - D \cdot X $ 和 $ \frac{dP}{dt} = v_p \cdot X - D \cdot P $。这里 $ \mu $ 和 $ v_p $ 是具体反应速率，它们通过相同的神经网络模型来建模。

4. **非参数化功能**：
   - **预处理函数**：$ f(c, x) $ 通常用于简化非参数化函数 $ g(\cdot) $ 的识别过程。在这个案例中，$ f(c, x) $ 被定义为 $[X, F, T]^T$，其中 $ X $ 是生物量，$ F $ 是进料速率，$ T $ 是培养温度。

5. **模型优化**：
   - **隐藏层神经元数量**：初步测试表明，隐藏层中五个神经元是最佳的，对应的参数维度为 $ dim(w) = 4 \times 5 + 6 \times 2 = 32 $。



### Bootstrap-Aggregated

这种Bootstrap聚集的方法特别适合于当数据集较小时，通过创建多个模型并聚合它们的预测来提高模型的鲁棒性和准确性。它允许模型捕捉到数据中的重要特征，同时通过聚合减少过拟合的风险。这种方法对于理解如何从 $g$ 到 $r$ 再到 $c$ 的变异性如何传播也提供了有价值的见解。在实际应用中，选择合适数量的 $n_{\text{boot}}$​ 和进行有效的参数初始化和优化是实现高性能模型的关键。

<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 15.17.04.png" alt="CleanShot 2024-05-15 at 15.17.04" style="zoom:67%;" />

#### 步骤 1: 数据重采样
- 数据被划分为三个集合：训练集（占总数据的一半），验证集（占四分之一），和测试集（占四分之一）。

- 训练集和验证集数据通过随机重采样 $n_{\text{boot}}$ 次从均匀分布中选取，形成多个训练/验证分区。重采样是以实验为单位进行的，不是观测值，以确保验证集不重复选择。

  ```python
  import numpy as np
  import pandas as pd
  from math import factorial
  from itertools import combinations
  
  def calculate_max_boot(n_tr, n_vd):
      """计算最大的bootstrap样本数"""
      return factorial(n_tr + n_vd) // (factorial(n_vd) * factorial(n_tr))
  
  def bootstrap_resampling(data, test_set, n_boot):
      # 假设data是一个DataFrame，其中每一行代表一个实验的观测数据
      # 分离测试集实验
      training_validation_data = data[~data['experiment_id'].isin(test_set)]
      
      # 获取训练/验证实验的唯一标识符
      experiments = training_validation_data['experiment_id'].unique()
      n_experiments = len(experiments)
      
      # 计算训练集和验证集的大小
      n_train = int(0.5 * n_experiments)  # 或其他适当的比例
      n_val = n_experiments - n_train
  
      # 计算最大的bootstrap样本数
      max_boot = calculate_max_boot(n_train, n_val)
      n_boot = min(n_boot, max_boot)  # 确保不超过最大样本数
  
      # 随机选择训练集和验证集的实验，确保不重复选择
      bootstrap_samples = []
      all_combinations = list(combinations(experiments, n_train))
      selected_combinations = np.random.choice(len(all_combinations), size=n_boot, replace=False)
  
      for idx in selected_combinations:
          train_experiments = all_combinations[idx]
          val_experiments = [exp for exp in experiments if exp not in train_experiments]
          
          # 获取训练和验证数据
          train_data = training_validation_data[training_validation_data['experiment_id'].isin(train_experiments)]
          val_data = training_validation_data[training_validation_data['experiment_id'].isin(val_experiments)]
          
          bootstrap_samples.append((train_data, val_data))
      
      return bootstrap_samples
  
  # 假设df是包含所有数据的DataFrame，其中包括一个名为'experiment_id'的列，标识每次实验
  # 假设已知的测试集实验ID列表
  test_set_ids = [101, 102, 103]  # 这应由您的数据集确定
  # n_boot表示要生成的重采样数据集对的数量
  bootstrap_datasets = bootstrap_resampling(df, test_set_ids, n_boot=100)
  
  # 打印第一对重采样数据集的信息
  print("Example of one bootstrap sample:")
  print("Training set:")
  print(bootstrap_datasets[0][0].head())
  print("Validation set:")
  print(bootstrap_datasets[0][1].head())
  ```

  1. **测试集保持不变**：将测试集从总数据集中分离出来，并保证在生成训练集和验证集的过程中测试集不被触及。
  2. **正确计算 $ n_{\text{boot}} $ 的最大数量**：增加了一个函数 `calculate_max_boot`，用来计算根据组合公式得到的训练集和验证集的最大可能组合数。
  3. **随机选择训练和验证集的过程中，不超过最大样本数**：通过 `itertools.combinations` 生成所有可能的组合，并从中随机选择需要的样本数，以确保不重复选择相同的验证集。


#### 步骤 2: 参数估计和模型验证
- 每一个训练/验证样本都会开发出一个不同的混合模型，而模型结构（如神经网络中隐藏节点的数量）保持不变，仅网络参数值 $w$ 允许变化。

- 这些参数值在高斯分布的 [-0.01, 0.01] 范围内随机初始化。

- 参数通过最小化加权最小二乘（WMSE）损失函数来估计，并采用基于梯度的方法（如Levenberg-Marquardt算法）优化。

  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/image-20240515001315944.png" alt="image-20240515001315944" style="zoom:50%;" />

-  The WMSE loss function is also monitored for the validation partition comprising  n_vd  experiments. The decision to stop the parameter estimation is made by when the validation WMSE starts to increase to avoid modeling measurement noise. 

- 参数估计重复100次(epochs)，从中选出表现最佳的模型用于后续聚合。

  

  ```python
  def train_model(train_data, val_data, epochs=100):
      input_size = train_data.shape[1] - 1  # Assuming last column is the target
      hidden_size = 5  # Example: 5 neurons in hidden layer
      output_size = 1  # Assuming single target variable
      weight = torch.tensor([1.0])  # Example weight
      # Create instance of the model
      model = SimpleNN(input_size, hidden_size, output_size)
      model.apply(init_weights)
  
      optimizer = torch.optim.Adam(model.parameters())  # Using Adam optimizer
      best_val_loss = float('inf')
      best_model = None
  
      for epoch in range(epochs):
          model.train()
          for batch in train_data:
              inputs, targets = batch[:, :-1], batch[:, -1]
              optimizer.zero_grad()
              outputs = model(inputs)
              loss = wmse_loss(outputs, targets, weight)
              loss.backward()
              optimizer.step()
  
          # Validate the model
          model.eval()
          with torch.no_grad():
              val_loss = 0
              for batch in val_data:
                  inputs, targets = batch[:, :-1], batch[:, -1]
                  outputs = model(inputs)
                  val_loss += wmse_loss(outputs, targets, weight).item()
              val_loss /= len(val_data)
  
          # Early stopping
          if val_loss < best_val_loss:
              best_val_loss = val_loss
              best_model = model.state_dict()  # Save the best model
          else:
              # Stop training if validation loss starts to increase
              break
  
      return best_model
  ```

#### 步骤 3: 聚合
- 聚合阶段是将 $n_{\text{boot}}$ 个混合模型通过平均其输出变量来聚合。

  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/image-20240515003056588.png" alt="image-20240515003056588" style="zoom:33%;" />

- 实际上，只有表现最佳的 $n$ 个模型被聚合，这里的 $n$ 是一个设计参数。

  -  The models are ranked according to their joint training-validation WMSE.

- 预测的浓度在给定时间 $t$ 计算为这 $n$ 个最佳模型的浓度平均值，相应的时间依赖性预测标准偏差也可以计算出来。

  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/image-20240515003114135.png" alt="image-20240515003114135" style="zoom:50%;" />

为了实现上述描述的聚合阶段，我们首先需要确保有一个方法来存储每个模型的训练和验证性能，以便于之后进行排序和选择最佳模型。然后，我们可以实现一个函数来计算这些最佳模型输出的平均值和标准偏差。

首先，我们需要假设我们已经有了一个列表，其中包含了模型的状态和它们对应的训练及验证集损失。然后我们实现聚合功能。

```python
import torch
import numpy as np

def aggregate_models(model_list, n_best, input_data):
    """
    聚合最佳模型的输出，使用训练集和验证集的联合WMSE进行排序
    :param model_list: 包含模型状态、训练损失和验证损失的元组列表 [(model_state_dict, train_loss, val_loss), ...]
    :param n_best: 要聚合的最佳模型数量
    :param input_data: 用于预测的输入数据，假设是一个Tensor
    :return: 平均预测和标准偏差
    """
    # 计算联合WMSE并排序，假设训练和验证损失有相同的权重
    sorted_models = sorted(model_list, key=lambda x: (x[1] + x[2]) / 2)[:n_best]

    # 载入模型并进行预测
    predictions = []
    for model_state, _, _ in sorted_models:
        model = SimpleNN(input_data.shape[1], 5, 1)  # 假设模型结构与之前定义相同
        model.load_state_dict(model_state)
        model.eval()
        with torch.no_grad():
            output = model(input_data).numpy()  # 进行预测并转换为NumPy数组
        predictions.append(output)

    # 计算预测值的平均值和标准偏差
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    return mean_predictions, std_predictions

# 示例使用
# 假设model_states包含了若干个模型的状态和它们的验证损失
# input_data是用于预测的输入数据
mean_preds, std_preds = aggregate_models(model_states, 5, torch.tensor(input_data))
```

1. **排序和选择**：函数`aggregate_models`首先根据传入的模型列表（包含模型状态和相应的训练，验证损失）进行排序，选择训练集和验证集损失的平均值最小的前`n_best`个模型。
2. **模型预测**：对每个选中的模型，加载其状态并对输入数据`input_data`进行预测。
3. **计算平均值和标准偏差**：收集所有选中模型的预测结果，计算这些预测结果的平均值和标准偏差。







### 过程数据分析总结

**图 2** 展示了使用 Doehlert 设计（DD）、中心复合设计（CCD）、和 Box-Behnken 设计（BBD）所得到的过程响应的变异性。

<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 10.00.14.png" alt="CleanShot 2024-05-15 at 10.00.14" style="zoom:50%;" />

- **生物量时间轮廓（图 2a）**：
  - DD和CCD的生物量平均浓度及其上下限（±𝜎）表现出显著的一致性。
  - BBD虽然平均响应类似，但由于探索了更极端的因素值（特别是在诱导时的生物量浓度方面），其分散性明显更高。

- **底物浓度时间轮廓（图 2b）**：
  - 不同设计之间的差异更为明显，这是因为在探索的低浓度范围内，底物浓度对设计因素的敏感性远大于其他状态变量。
  - 尽管如此，三种设计探索的区域仍然较为一致。
  - 更复杂的突变是由不同时间点的生物量诱导扰动引起的。
  - BBD探索了略微更广泛的区域。

- **产品时间轮廓（图 2c）**：
  - DD和CCD探索的空间与之前类似。
  - BBD在此明显不同，因为它探索了更广泛的空间，并且与其他设计相比，其产品浓度相对较低。

**数据详情**可在附加文件A中查看。

这一分析显示了不同实验设计对过程变量响应的影响，有助于理解各设计策略的适用性和效果。





### 数据重采样的影响

本研究首先将混合引导聚集方法应用于 Doehlert 数据集。具体的实验和数据处理过程如下：

<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 11.06.23.png" alt="CleanShot 2024-05-15 at 11.06.23" style="zoom:50%;" />

- **测试实验的选择**：为了进行测试，选择了三个实验（编号为13、14和15），这些实验包括一个中心点和两个在所有三个设计因素上变化极大的极端实验。这些测试实验在整个研究中始终保持不变。
- **训练和验证实验的选择**：前12个实验被选作训练和验证重采样的数据源，其中十个实验用于训练，两个实验用于验证。
- **重采样过程**：进行了14次重采样，每次重采样生成了不同的十个训练实验和两个验证实验的组合。
- **模型开发**：每个重采样数据集都用来开发一个结构和大小相同的混合模型。
- **模型误差**：
  - **训练/验证 WMSE**：模型的训练/验证加权均方误差（WMSE）在0.0062（模型2）到0.0835（模型12）之间变化，表现出约13倍的变化幅度。
  - **测试 WMSE**：测试 WMSE 在0.0056（模型1）到0.1630（模型12）之间变化，有29倍的变化幅度。
- **模型评估**：
  - **最佳模型**：模型2是最佳模型，具有最低的综合训练/验证/测试误差。
  - **最差模型**：模型12是表现最差的模型。



### 聚合的影响

- **聚合数量的优化**：
  
  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 11.07.07.png" alt="CleanShot 2024-05-15 at 11.07.07" style="zoom:50%;" />

  - 根据图3的展示，最佳的聚合数量为 n = 4，此时聚合的混合模型显示出最小的 WMSE。
  - 当 n = 4 时，聚合的模型不仅降低了训练/验证 WMSE（相比最佳单一模型降低了16.1%），更重要的是，它也将测试 WMSE 降低了38%。
  
- **聚合数量超过最佳值的影响**：
  - 图3还显示，当 n > 10 时，WMSE 急剧增加。这表明聚合了一些具有非常大误差的混合模型，这些模型可能被视为应该从分析中剔除的“异常”模型。

- **模型选择的重要性**：
  - 在bootstrap聚集框架中，**去除表现最差的模型非常重**要。从采样的角度来看，这也是有道理的，因为基于**非常不同的过程条件**或表现出与其他实验**非常不同的行为的样本（实验）**，可能会阻碍训练收敛到最佳总体性能。在包含这些样本的验证集中，可能会出现这种情况。
  - 对于通过统计设计实验获得的数据而言，设计的边缘可能包含一些不易被混合模型外推的极端条件。



### **Mean and variance of dynamical profiles**

Bootstrap聚集（Bagging）的一个优势是在动态系统中自动直接计算预测误差边界。这种方法不仅提供预测值（均值），还提供预测误差界限（均值周围的±𝜎）随时间变化的情况（见公式 4-5）。下面是针对DD设计中聚合了 n = 4 的混合模型（前文讨论过）的案例进行说明。

- **生物量和产品浓度预测**：
  
  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 11.30.17.png" alt="CleanShot 2024-05-15 at 11.30.17" style="zoom:50%;" />

  - **图 4** 展示了 DD 训练集中两个实验的生物量和产品浓度预测以及相应的特定速率。
  - 生物量和产品浓度的速率和浓度剖面清晰地显示了一旦达到预定义的诱导生物量浓度 $X_{\text{ind}}$，**生长阶段和生产阶段之间的转换**。
  
- **误差界限分析**：
  
  - 特定生物量**速率的误差界限**在**生长阶段比生产阶段更大**。
  - 尽管特定速率在某些部分有显著变化，**但相应的浓度预测仅显示出轻微的变化**。
  
- **误差传播和减震效应**：
  
  - 这些观察结果部分可以通过物质平衡的整合对误差传播的减震效应来解释，**这也意味着速率的轻微差异不会导致浓度的重大偏差**。
  - 在诱导时生物量浓度较大的实验中，可以看到：
    1. 实验结束时，**浓度（生物量和产品）的误差界限变得更宽**。
    2. 特定生物量**生长**和特定**产品形成的速率**误差界限分别**保持恒定和增长**。
  
- **产品浓度预测**：
  
  - 值得注意的是，特定产品形成速率与预测的生物量浓度的乘积可能会导致产品浓度预测的显著变化，然而实际并非如此。这可能部分是由于整合的影响。



### 混合Bootstrap聚集是否有益？

为了更有信心回答这个问题，混合引导聚集框架被应用于三种不同的数据集设计（DD、CCD和BBD）。这些设计产生了不同的数据集，因此导致了不同的模型和不同的聚合结果。

- **实验数量和分配**：
  - DD和BBD的实验总数为15个，CCD为17个。
  - 在所有情况下，选取了10个实验用于训练，2个实验用于验证。
  - 对于DD和BBD设计，最后三个实验（13-15号）被选为测试集；而对于CCD，最后五个实验（13-17号）被选为测试集。

- **重采样和聚合数量**：
  - 在所有案例中重采样事件的数量都保持相同（nboot = 14）。
  - 最优聚合数量的调查与之前对DD设计的方法相同，确定最佳模型聚合数量分别为DD的4个，CCD的3个，以及BBD的3个。

- **聚合模型与单一最佳模型的性能比较**（见表3）：

  <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 15.23.49.png" alt="CleanShot 2024-05-15 at 15.23.49" style="zoom:50%;" />

  - 引导聚集方法在三种设计中都改善了结果。其中 CCD 使用了 5 个子模型, BBD 使用了 3 个子模型, DD 使用了 4个子模型
  - 在训练/验证分区中，改善效果较不显著。聚合模型相对于最佳单一模型的WMSE减少了16.1%、9.1%和2.0%，分别对应DD、BBD和CCD设计。
  - 在测试数据集中，改善尤为显著，聚合模型相对于最佳单一模型的WMSE减少了38.1%、51.6%和40.0%，分别对应DD、BBD和CCD设计。

- **意义**：
  - 测试分区中WMSE的大幅减少尤其重要，因为这证明了最终混合模型预测开发模型之外的新实验的能力。

- 其中提到的“**最佳单一模型**”指的是在多次重采样和模型训练过程中，表现最优的单个模型。这个模型是通过比较所有单独训练的模型在验证集或测试集上的性能得到的。下面是如何得到最佳单一模型的具体过程：

    - **性能对比**：比较所有训练模型在验证集上的性能指标，选择WMSE最低的模型作为最佳单一模型。

    - **测试集验证**：为了验证最佳单一模型的泛化能力，还需要在独立的测试集上评估该模型。测试集的结果通常用来最终确认模型的有效性和稳定性。

    - 它提供了一个性能基准，用于评估聚合模型是否真正提供了性能上的改进。

    - 它可以帮助理解不同重采样数据集对模型性能的影响，从而更好地理解数据的变异性对预测性能的影响。

    - 在一些应用场景中，最佳单一模型本身就可能足够好，无需进一步的聚合。




### 整体预测能力

**图 5** 展示了使用表 3 中的混合模型对三个数据集中的生物量和产品的预测值与实测值的比较。预测误差界限计算出来，并以黑色条形显示。在三个数据集的训练/验证和测试分区中，生物量和产品的预测值与实验值非常吻合（少数例外情况下讨论）。测试集的预测误差略高，但与训练-验证集的误差相当。

<img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 15.31.07.png" alt="CleanShot 2024-05-15 at 15.31.07" style="zoom:50%;" />

- **BBD设计**：在测试数据集中，**BBD的预测误差幅度是三种设计中最高的**，这可能由其设计的性质解释。**BBD探索了极端实验，覆盖了较广的空间**（见图 2），但这使得**难以获得一个能描述整个空间行为的模型**。在这种情况下，模型的聚合原则上可以改善整个空间的模型性能，实际上也的确如此，测试数据集的WMSE减少了两倍。这表明，**设计越极端，引导聚集可能越有益**。此外，在高生物量浓度区域，误差界限较高（见图 4b）。这些生物量值来自于在非常不同条件下进行的三个不同实验。可能是因为单一混合模型没有捕捉到系统的整体行为，由于这些实验每一个在训练和验证集中的存在变化，导致了更大的误差界限。然而，聚合模型（平均值）似乎很好地捕捉了系统的行为。

- **CCD设计**：在一个展示低产品浓度的测试数据实验中，CCD模型略微过预测（见图 3a）。该实验在最低设计温度下进行，因此，聚合模型进行了外推，这是已知会削弱预测能力的。

- **DD设计**：在测试集中包含的一个实验产生了最高的产品浓度，**聚合的混合模型非常好地预测了单一最高产品浓度点**（见图 3c）。这一结果显示，该模型对于**高于训练-验证数据观察到的产品浓度具有预测能力**，这在旨在最大化最终产品滴度时是一个期望的特征。



### 识别最优工艺

实验设计方法常用于工艺优化，其目的是在探索的过程条件中找到最优点（即工艺最优的插值）。因此，比较表3中三个聚合混合模型对最终产品浓度的预测变得很有意义。

- **图 6** 显示了在培养时间17小时时的“真实”最终产品浓度，作为探索设计空间的函数（图 6a），以及由CCD、BBD和DD数据集衍生的bootstrap聚合混合模型的相应预测（图 6b-d）。可以观察到，**所有三个聚合混合模型都正确地指示了可以找到最高产品浓度的过程区域**。然而，这一**区域的形状和预测浓度的准确性只有在CCD和DD的引导聚合混合模型（BAHM）中得到了良好的保留。**
  - **CCD-BAHM在整个空间内最好地描述了产品浓度**，这是预期的，因为空间已经通过实验得到了良好的表征。
  - **DD-BAHM似乎**比其他设计更好地描述了过程在调查范围极限的行为，而且在**整个空间的预测也是良好的**。DD探索空间的方式似乎帮助混合模型学习了系统的行为，这也与其他研究者对其他建模技术的发现相符合。

有时，最优工艺可能位于**探索设计空间之外**。在这里研究的问题中，**最大化最终产品滴度的条件位于探索的过程区域之外**，即在更高的生物量诱导浓度（**最优为25** g/kg，**研究范围5-19** g/kg）。**这样做是为了评估混合建模框架的外推能力**。研究这些变化的影响似乎特别有趣，因为一方面生物量浓度的预测错误会导致所有其他化合物的预测错误（它与特定速率相乘，因此影响所有浓度的演变），但另一方面，它也是非参数模型的输入（通常不擅长外推）。

- **图 7** 显示了三个聚合混合模型的预测和最终产品浓度在最优生物量诱导浓度（25 g/kg）的真实反应表面。可以看到，所有三个模型的预测反应表面与真实反应表面相当吻合。

    <img src="Review-Datahow-2019b-Ensembles_Hybrid-Model_Continuous-Propagation-Model-NN.assets/CleanShot 2024-05-15 at 15.42.35.png" alt="CleanShot 2024-05-15 at 15.42.35" style="zoom:50%;" />

    除了BBD-BAHM外，它表明最优条件位于调查范围之外（对于温度和进料率），其他模型相当准确地捕捉了最优条件。误差界限提供了对预测可靠性的指示，可以看出CCD-BAHM提供了最可靠的预测，其次是DD-BAHM。



### 结论

本研究探讨了一种混合建模方法，旨在从数据中提取知识。特别是，研究了一种bootstrap聚集的混合建模框架，用以减少由训练和验证数据选择带来的偏差，这种偏差在小数据集中特别明显，**尤其是当数据来自于统计设计的实验且条件变化明显时**。

- **数据集与方法比较**：
  - 利用三种不同设计（中心复合设计、Box-Behnken设计和Doe.hlert设计）生成的三个大肠杆菌连续投料过程的合成数据集，对比了不同方法。
  
- **主要发现**：
  - 结果表明，当数据源自统计设计的实验时，提出的bootstrap聚集框架显著增强了混合半参数模型的预测能力。
  - 在生物过程开发的背景下，这一优势尤为重要，因为混合模型可以在每个过程开发阶段更准确地预测最佳操作条件，从而在全球范围内减少过程开发的实验工作量。

- **动态系统误差界限计算的应用**：
  - 能够轻松计算动态系统及其不同部分的可靠误差界限，这对于过程监控和过程优化/控制尤其有趣。
  - 在线决策不仅可以基于利润函数，还可以基于预测的可靠性量化度量。

- **潜在缺点**：
  - 当然，主要的缺点是增加了计算时间，大约是未进行引导聚集时的nboot倍（在本研究中为14倍）。
  - 然而，随着计算能力的不断增强，这一缺点在实践中并不被视为严重限制。

- **未来研究方向**：
  - 未来应研究是否其他聚合方法，如特定于数据域的模型预测加权（类似堆叠或提升），能进一步改善预测性能。
