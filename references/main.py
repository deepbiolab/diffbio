import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
import pyDOE3
import GPy
from emukit.model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper

# ==========================================
# 1. 虚拟生物过程环境 (保持不变)
# ==========================================

class InSilicoBioprocess:
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.n_substrates = 8
        self.noise_level = 0.03  # 3% measurement noise

        # ==========================================
        # [CORRECTION] Parameters from Table S1
        # ==========================================
        # Global Parameters
        self.mu_max_base = 0.8    # Unit: 1/h
        self.d_base = 0.001       # Death rate, Unit: 1/h
        self.X0 = 0.01            # Start biomass, Unit: g/L

        # Substrate Specific Parameters (A -> H)
        # Ks: Monod constant (g/L)
        self.Ks_base = np.array([0.213, 0.260, 0.390, 0.355, 0.356, 0.422, 0.016, 0.060])
        
        # Ki: Inhibition constant (g/L)
        self.Ki_base = np.array([160.0, 18.0, 38.0, 12.0, 56.0, 10.6, 16.0, 42.0])
        
        # Yx/s: Yield coefficient (g/g)
        self.Yxs_base = np.array([0.09, 0.11, 0.14, 0.21, 0.27, 0.27, 0.33, 0.44])
        
        # Apply random perturbation (+/- 33%) to generate specific scenario
        self.randomize_parameters()

    def randomize_parameters(self):
        """
        Paper: "each of the displayed values was randomly increased or decreased 
        by up to ± 33% for each scenario."
        """
        perturb = lambda x: x * (1 + self.rng.uniform(-0.33, 0.33))
        
        self.mu_max = perturb(self.mu_max_base)
        self.d = perturb(self.d_base) # perturb death rate too
        self.Ks = perturb(self.Ks_base)
        self.Ki = perturb(self.Ki_base)
        self.Yxs = perturb(self.Yxs_base)

    def _growth_rate(self, S):
        """
        Calculates specific growth rate (mu) based on Extended Monod Model.
        Equation: mu = mu_max * Product( S_i / (Ks_i + S_i + S_i^2/Ki_i) )
        """
        # Extended Monod term with inhibition
        limitations = S / (self.Ks + S + (S**2 / self.Ki))
        
        # Ensure non-negative (numerical stability)
        limitations = np.maximum(limitations, 0)
        
        # Multiplicative limitation assumption
        mu = self.mu_max * np.prod(limitations)
        return mu

    def _ode_system(self, y, t):
        X = y[0]       # Biomass
        S = y[1:]      # Substrates (8 dims)
        
        mu = self._growth_rate(S)
        
        # [CORRECTION] Added death rate term: dX/dt = (mu - d) * X
        dXdt = (mu - self.d) * X
        
        # dS/dt = - (1/Yxs) * (mu * X) 
        # Note: Substrate consumption depends on growth (mu), not net accumulation (mu-d)
        # Usually consumption is linked to gross growth rate.
        dSdt = - (1.0 / self.Yxs) * (mu * X)
        
        return np.concatenate(([dXdt], dSdt))

    def evaluate(self, conditions):
        """
        Input: (N, 8) normalized conditions [0, 1]
        Output: (N, 1) final biomass
        """
        results = []
        # Assuming stock solution max concentrations based on context, 
        # here we set an arbitrary scaling factor or assume 1.0 means 'max reasonable concentration'.
        # In paper simulations, they optimized specific amounts. 
        # Let's assume max_concentration = 5.0 g/L for simplicity to match scale of Ks/Ki
        max_concentration = 5.0 
        
        for condition in conditions:
            # Map [0,1] input to physical concentration
            S0 = condition * max_concentration
            
            y0 = np.concatenate(([self.X0], S0))
            t = np.linspace(0, 96, 100) # 96 hours cultivation
            
            # Solve ODE
            sol = odeint(self._ode_system, y0, t)
            
            # Get final biomass
            final_biomass = sol[-1, 0]
            
            # Add Gaussian noise (3%)
            noise = self.rng.normal(0, self.noise_level * final_biomass)
            results.append(max(0, final_biomass + noise))
            
        return np.array(results).reshape(-1, 1)

# ==========================================
# 2. 优化算法实现 (集成 PyDOE3 和 Emukit Batch)
# ==========================================

# --- Benchmark A: DoE (利用 pyDOE3) ---
def run_doe_strategy(env, n_experiments=96):
    """
    Step 1: Screening (LHS Design via PyDOE3)
    Step 2: Optimization (Top factors exploitation)
    """
    n_screening = 48
    n_factors = 8
    
    # [NEW] 使用 PyDOE3 生成拉丁超立方设计 (LHS)
    # criterion='center' 意味着取每个区间的中心点，分布更均匀
    X_screen = pyDOE3.lhs(n_factors, samples=n_screening, criterion='center')
    Y_screen = env.evaluate(X_screen)
    
    # 简单的线性筛选
    model = LinearRegression()
    model.fit(X_screen, Y_screen)
    coeffs = model.coef_.flatten()
    
    # Step 2: Optimization
    # 选取系数最大的 3 个因子进行强化
    top_indices = np.argsort(coeffs)[-3:] 
    
    X_opt = np.zeros((48, n_factors))
    for i in range(n_factors):
        if i not in top_indices:
            # 不显著因子固定在低位
            X_opt[:, i] = 0.1 
        else:
            # 显著因子在高位探索 (再次利用 LHS 保证覆盖)
            # 生成 48 个样本，缩放到 [0.5, 1.0] 区间
            lhs_samples = pyDOE3.lhs(1, samples=48).flatten()
            X_opt[:, i] = 0.5 + 0.5 * lhs_samples
            
    Y_opt = env.evaluate(X_opt)
    
    return np.max(np.vstack((Y_screen, Y_opt)))

# --- Benchmark B: BBO (利用 Emukit Batch Loop) ---
def run_bbo_strategy(env, n_init=48, batch_size=48):
    """
    手动实现 Batch BO (Kriging Believer 策略)
    替代 Emukit 内置的 LocalPenalization，以解决库缺失和 Scipy 维度报错问题。
    """
    # 1. 定义空间
    space = ParameterSpace([ContinuousParameter(f'S{i}', 0, 1) for i in range(8)])
    
    # 2. 初始设计 (使用 pyDOE3 LHS)
    X_init = pyDOE3.lhs(8, samples=n_init, criterion='center')
    Y_init = env.evaluate(X_init)
    
    # 3. 构建 GP 模型
    # 使用 Matern52 核函数，处理高维生物数据更平滑
    kern = GPy.kern.Matern52(input_dim=8, ARD=True)
    gpy_model = GPy.models.GPRegression(X_init, Y_init, kern, noise_var=1e-5)
    gpy_model.optimize()
    model = GPyModelWrapper(gpy_model)
    
    # 4. 手动生成 Batch (Kriging Believer Loop)
    X_batch = []
    
    # 备份原始数据，因为我们在循环中要"篡改"模型数据
    X_orig = model.model.X.copy()
    Y_orig = model.model.Y.copy()
    
    # 使用 NLCB (符合论文 Table S3)
    acquisition = NegativeLowerConfidenceBound(model, beta=1.96)
    optimizer = GradientAcquisitionOptimizer(space)
    
    print(f"  > Generating batch of {batch_size} points using Kriging Believer...")
    
    for i in range(batch_size):
        # a. 找到当前最有潜力的点
        # x_new 的 shape 已经是 (1, 8)
        x_new, _ = optimizer.optimize(acquisition)
        X_batch.append(x_new)
        
        # b. "幻觉"预测 (Predict): 获取该点的预测均值
        # [FIX]: 直接传入 x_new，不要再包裹 np.array([])
        mu, _ = model.predict(x_new)
        
        # c. "篡改"数据 (Update): 把这个假数据加入模型
        # 这样模型会认为这个地方已经探索过了，方差变为0，下次就不会再选这里
        X_temp = np.vstack([model.model.X, x_new])
        Y_temp = np.vstack([model.model.Y, mu])
        
        # 更新模型数据 (但不重新优化超参数，为了速度)
        model.set_data(X_temp, Y_temp)
    
    # 5. 恢复原始模型并评估真实的 Batch
    # 还原数据，清除幻觉
    model.set_data(X_orig, Y_orig)
    
    # 转换为 numpy array (48, 8)
    X_new_batch = np.vstack(X_batch)
    
    # 在虚拟环境评估这 48 个点
    Y_new_batch = env.evaluate(X_new_batch)
    
    # 6. 返回结果 (初始数据 + 新 Batch 中的最大值)
    all_Y = np.vstack((Y_init, Y_new_batch))
    return np.max(all_Y)

# --- 你的算法 (占位符) ---
def run_my_algorithm(env):
    # 这里也可以用 pyDOE3 做基线
    X_rand = pyDOE3.lhs(8, samples=96)
    Y_rand = env.evaluate(X_rand)
    return np.max(Y_rand)

# ==========================================
# 3. 运行对比实验 (10 Scenarios)
# ==========================================

n_scenarios = 10
results_doe = []
results_bbo = []
results_my = []

print(f"Starting In Silico Benchmark using PyDOE3 & Emukit Batch...")
print(f"{'Scenario':<10} | {'DoE':<10} | {'BBO':<10} | {'My Algo':<10}")
print("-" * 46)

for seed in range(n_scenarios):
    env = InSilicoBioprocess(seed=seed)
    
    best_doe = run_doe_strategy(env)
    best_bbo = run_bbo_strategy(env)
    best_my  = run_my_algorithm(env)
    
    results_doe.append(best_doe)
    results_bbo.append(best_bbo)
    results_my.append(best_my)
    
    print(f"{seed+1:<10} | {best_doe:.4f}     | {best_bbo:.4f}     | {best_my:.4f}")

# ==========================================
# 4. 绘图 (论文同款 Paired Plot)
# ==========================================
plt.figure(figsize=(10, 6))

x_coords = [1, 2, 3]
labels = ['DoE\n(PyDOE3 LHS)', 'BBO\n(Emukit Batch)', 'My Method']

# 绘制 Paired Lines
for i in range(n_scenarios):
    y_values = [results_doe[i], results_bbo[i], results_my[i]]
    # 灰色连线
    plt.plot(x_coords, y_values, color='gray', alpha=0.6, linestyle='-', marker='o')
    
    # 标记赢家 (绿色)
    if results_bbo[i] > results_doe[i]:
         plt.plot(2, results_bbo[i], 'g.', markersize=10)

plt.xticks(x_coords, labels, fontsize=12)
plt.ylabel('Maximum Biomass (g/L)', fontsize=12)
plt.title(f'In Silico Comparison (10 Random Scenarios)\nBatch Size = 48', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 统计分析
avg_doe = np.mean(results_doe)
avg_bbo = np.mean(results_bbo)
improvement = (avg_bbo - avg_doe) / avg_doe * 100

text_str = f"Avg Improvement (BBO vs DoE): +{improvement:.2f}%"
plt.text(1.5, min(results_doe)*1.1, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.show()