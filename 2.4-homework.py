import numpy as np

# 生成示例数据
np.random.seed(0)
N, D = 100, 2
X = np.random.randn(N, D)
true_w = np.array([2.0, -3.0]) # 2, 1
true_b = 1.5
epsilon = np.random.laplace(0, 1, size=N)  # 拉普拉斯分布噪声
print(f"x.shape: {X.shape}, true_w.shape: {true_w.shape}")
y = X @ true_w + true_b + epsilon

# 初始化参数
w = np.zeros(D)
b = 0.0
lr = 0.01

# L1损失的梯度（次梯度)
'''
L1损失(L1 Loss): 也叫绝对值损失或曼哈顿距离.
L2损失(L2 Loss): 也叫平方损失或欧氏距离.

服从拉普拉斯分布 的负对数似然为 L1损失
'''
"""
L1损失函数在预测值等于真实值时 (ypred=y),不可微,梯度为0或在[-1,1]之间（次梯度).
如果多次采样到ypred=y,梯度为0,参数可能停在当前值而不再更新,导致收敛速度慢或停在非全局最优.
因此,L1损失下SGD在驻点附近可能“卡住”,收敛不如L2损失平滑.
"""
def l1_subgradient(y_true, y_pred, x):
    sign = np.sign(y_pred - y_true)  # 次梯度
    return sign * x, sign  # 对w和b的梯度

# 随机梯度下降
for epoch in range(100):
    idx = np.random.randint(N)  # 随机采样
    x_i, y_i = X[idx], y[idx]
    y_pred = np.dot(w, x_i) + b
    grad_w, grad_b = l1_subgradient(y_i, y_pred, x_i)
    w -= lr * grad_w
    b -= lr * grad_b

# 打印参数
print("Learned w:", w)
print("Learned b:", b)

# ------------------------------- 高斯分布 -------------------------------


import numpy as np

# 1. 生成模拟数据（带高斯噪声的一元线性回归）
np.random.seed(42)
N = 100  # 样本数量
X = np.linspace(0, 10, N).reshape(-1, 1)  # 100个点，特征为1维
true_w = 2.5
true_b = -1.0
sigma = 1.0  # 高斯噪声标准差
noise = np.random.normal(0, sigma, size=(N, 1))
y = true_w * X + true_b + noise  # 真实标签

# 2. 添加偏置项
X_bias = np.hstack([X, np.ones_like(X)])  # 形状 (N, 2)

# 3. 解析解（正规方程）
# w = (X^T X)^{-1} X^T y
theta_hat = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
# (2, 1)
print(f"theta_hat.shape: {theta_hat.shape}")

w_hat, b_hat = theta_hat[0, 0], theta_hat[1, 0]
print(f"解析解: w = {w_hat:.3f}, b = {b_hat:.3f}")

# 4. 梯度下降法
w, b = 0.0, 0.0
lr = 0.01
epochs = 30000

# L(w, b) = 1/N * sum((y_pred - y)^2)
# dL/dw = 2/N * sum((y_pred - y) * X)
for epoch in range(epochs):
    y_pred = w * X + b
    grad_w = np.mean((y_pred - y) * X)
    grad_b = np.mean(y_pred - y)
    w -= lr * grad_w
    b -= lr * grad_b

print(f"梯度下降：w = {w:.3f}, b = {b:.3f}")


