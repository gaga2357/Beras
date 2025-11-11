"""
阶段 2：激活函数
========================================

学习目标：
1. 理解激活函数的作用
2. 实现常用激活函数及其导数
3. 处理数值稳定性问题

关键概念：
- 非线性变换
- ReLU、Sigmoid、Softmax
- 数值稳定性（防止溢出）
"""

import numpy as np
from exercises.01_tensor import Tensor


# ==================== 激活函数 ====================

def relu(x):
    """
    ReLU 激活函数：f(x) = max(0, x)
    
    TODO: 任务 2.1
    
    前向传播：
    - out = max(0, x)
    
    反向传播：
    - df/dx = 1 if x > 0 else 0
    
    关键问题思考：
    1. x = 0 时导数是多少？（通常取 0 或 1）
    2. 为什么 ReLU 会导致"神经元死亡"？
    3. ReLU 相比 Sigmoid 的优势是什么？
    
    提示：
    - 使用 np.maximum(0, x.data)
    - 缓存 mask = (x.data > 0) 用于反向传播
    
    Args:
        x: Tensor, 输入
    
    Returns:
        Tensor, 输出
    """
    # TODO: 在这里实现
    pass


def sigmoid(x):
    """
    Sigmoid 激活函数：f(x) = 1 / (1 + e^(-x))
    
    TODO: 任务 2.2
    
    前向传播：
    - out = 1 / (1 + exp(-x))
    
    反向传播：
    - df/dx = f(x) * (1 - f(x))
    
    关键问题思考：
    1. 为什么 Sigmoid 会导致梯度消失？
    2. 如何避免 exp 溢出？（x 很大或很小时）
    3. Sigmoid 适合什么场景？
    
    提示：
    - 数值稳定性：
      if x >= 0: sigmoid = 1 / (1 + exp(-x))
      else: sigmoid = exp(x) / (1 + exp(x))
    - 可以利用 sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    
    Args:
        x: Tensor, 输入
    
    Returns:
        Tensor, 输出
    """
    # TODO: 在这里实现
    pass


def tanh(x):
    """
    Tanh 激活函数：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    TODO: 实现 Tanh（可选）
    
    反向传播：
    - df/dx = 1 - f(x)^2
    
    提示：
    - 可以用 numpy 的 np.tanh
    - tanh(x) = 2 * sigmoid(2x) - 1
    """
    # TODO: 在这里实现
    pass


def softmax(x, axis=-1):
    """
    Softmax 激活函数（用于多分类）
    
    TODO: 任务 2.3
    
    公式：
    softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    性质：
    - 输出是概率分布（和为 1）
    - 输出范围 [0, 1]
    
    反向传播（重要！）：
    softmax 的导数是雅可比矩阵：
    - df_i/dx_i = f_i * (1 - f_i)
    - df_i/dx_j = -f_i * f_j  (i ≠ j)
    
    关键问题思考：
    1. 如何避免 exp 溢出？
    2. 为什么要减去 max？
    3. Softmax 的导数为什么是矩阵？
    
    数值稳定技巧：
    softmax(x) = softmax(x - max(x))
    证明：
    exp(x_i - max) / Σ exp(x_j - max) 
    = exp(x_i) * exp(-max) / [Σ exp(x_j) * exp(-max)]
    = exp(x_i) / Σ exp(x_j)
    
    Args:
        x: Tensor, 形状 [batch_size, num_classes]
        axis: 归一化的轴
    
    Returns:
        Tensor, 形状同 x
    """
    # TODO: 在这里实现
    pass


def log_softmax(x, axis=-1):
    """
    Log Softmax：log(softmax(x))
    
    TODO: 实现 Log Softmax（可选，用于交叉熵）
    
    数值稳定：
    log_softmax(x) = x - max(x) - log(Σ exp(x - max(x)))
    
    提示：
    - 不要先算 softmax 再取 log（数值不稳定）
    - 直接用稳定的公式
    """
    # TODO: 在这里实现
    pass


# ==================== 测试代码 ====================

def test_relu():
    """测试 ReLU"""
    print("=" * 50)
    print("测试 1：ReLU 激活函数")
    print("=" * 50)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)
    
    print(f"输入: {x.data}")
    print(f"输出: {y.data}")
    print(f"期望: [0, 0, 0, 1, 2]")
    
    # 反向传播
    y.backward()
    print(f"梯度: {x.grad}")
    print(f"期望: [0, 0, 0, 1, 1]")
    print()


def test_sigmoid():
    """测试 Sigmoid"""
    print("=" * 50)
    print("测试 2：Sigmoid 激活函数")
    print("=" * 50)
    
    x = Tensor([0.0], requires_grad=True)
    y = sigmoid(x)
    
    print(f"输入: {x.data}")
    print(f"输出: {y.data}")
    print(f"期望: [0.5]")
    
    # 反向传播
    y.backward()
    print(f"梯度: {x.grad}")
    print(f"期望: sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25")
    
    # 测试数值稳定性
    x_large = Tensor([100.0], requires_grad=True)
    y_large = sigmoid(x_large)
    print(f"\n大值输入: {x_large.data}")
    print(f"输出: {y_large.data}（应该接近 1.0）")
    
    x_small = Tensor([-100.0], requires_grad=True)
    y_small = sigmoid(x_small)
    print(f"\n小值输入: {x_small.data}")
    print(f"输出: {y_small.data}（应该接近 0.0）")
    print()


def test_softmax():
    """测试 Softmax"""
    print("=" * 50)
    print("测试 3：Softmax 激活函数")
    print("=" * 50)
    
    # 简单测试
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = softmax(x)
    
    print(f"输入: {x.data}")
    print(f"输出: {y.data}")
    print(f"输出和: {y.data.sum()}（应该是 1.0）")
    
    # 反向传播
    y.backward()
    print(f"梯度: {x.grad}")
    
    # 测试数值稳定性
    x_large = Tensor([[1000.0, 1001.0, 1002.0]], requires_grad=True)
    y_large = softmax(x_large)
    print(f"\n大值输入: {x_large.data}")
    print(f"输出: {y_large.data}")
    print(f"输出和: {y_large.data.sum()}（应该是 1.0）")
    print()


def test_gradient_check():
    """梯度检查"""
    print("=" * 50)
    print("测试 4：梯度检查")
    print("=" * 50)
    
    def numerical_gradient(func, x, epsilon=1e-5):
        """计算数值梯度"""
        grad = np.zeros_like(x.data)
        for i in range(x.data.size):
            x_plus = x.data.copy()
            x_minus = x.data.copy()
            
            x_plus.flat[i] += epsilon
            x_minus.flat[i] -= epsilon
            
            y_plus = func(Tensor(x_plus))
            y_minus = func(Tensor(x_minus))
            
            grad.flat[i] = (y_plus.data - y_minus.data).sum() / (2 * epsilon)
        
        return grad
    
    # 测试 ReLU
    x = Tensor([1.0, -1.0, 0.0], requires_grad=True)
    y = relu(x)
    y.backward()
    
    numerical_grad = numerical_gradient(relu, x)
    
    print("ReLU 梯度检查:")
    print(f"自动微分: {x.grad}")
    print(f"数值梯度: {numerical_grad}")
    print(f"误差: {np.abs(x.grad - numerical_grad).max()}")
    
    if np.abs(x.grad - numerical_grad).max() < 1e-5:
        print("✅ ReLU 梯度检查通过！")
    else:
        print("❌ ReLU 梯度检查失败！")
    print()


if __name__ == "__main__":
    print("\n")
    print("🐻 Bears 学习之旅 - 阶段 2：激活函数")
    print("\n")
    
    # TODO: 完成激活函数的实现后，运行以下测试
    # test_relu()
    # test_sigmoid()
    # test_softmax()
    # test_gradient_check()
    
    print("\n")
    print("💡 提示：")
    print("1. 先实现 ReLU（最简单）")
    print("2. 再实现 Sigmoid（注意数值稳定性）")
    print("3. 最后实现 Softmax（最复杂，涉及归一化）")
    print("4. 使用梯度检查验证实现")
    print("5. 思考：为什么深度学习中常用 ReLU？")
    print("\n")
