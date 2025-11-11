"""
阶段 1：计算图与自动微分
========================================

学习目标：
1. 理解计算图的概念
2. 实现支持自动微分的 Tensor 类
3. 掌握反向传播的原理

关键概念：
- 计算图（Computational Graph）
- 自动微分（Automatic Differentiation）
- 链式法则（Chain Rule）
- 反向传播（Backpropagation）
"""

import numpy as np
from typing import Optional, List


class Tensor:
    """
    支持自动微分的张量类
    
    关键属性：
    - data: 存储数据的 numpy 数组
    - grad: 存储梯度
    - requires_grad: 是否需要计算梯度
    - grad_fn: 生成该张量的操作（用于构建计算图）
    - _prev: 父节点（计算图中的依赖）
    """
    
    def __init__(self, data, requires_grad=False):
        """
        初始化 Tensor
        
        TODO: 任务 1.1
        - 将输入数据转换为 numpy array（类型为 float32）
        - 初始化 grad 为 None
        - 设置 requires_grad
        - 初始化 grad_fn 为 None（用于记录操作）
        - 初始化 _prev 为空集合（用于存储父节点）
        
        提示：
        - 考虑输入可能是 list, int, float, np.ndarray 等类型
        - grad 应该和 data 形状相同，但一开始是 None
        """
        # TODO: 在这里实现
        pass
    
    def backward(self, grad=None):
        """
        反向传播，计算梯度
        
        TODO: 任务 1.5
        - 如果 grad 为 None，默认为全 1（标量情况）
        - 累加梯度到 self.grad
        - 调用 grad_fn 计算父节点梯度
        - 递归调用父节点的 backward
        
        关键问题思考：
        1. 为什么梯度要累加而不是覆盖？
        2. 如何避免重复计算？
        3. 如何处理多个子节点的梯度？
        
        提示：
        - 使用拓扑排序确保正确的计算顺序
        - 梯度累加：self.grad = self.grad + grad（如果已存在）
        """
        # TODO: 在这里实现
        pass
    
    def zero_grad(self):
        """
        梯度清零
        
        TODO: 将 self.grad 设置为 None 或 全零数组
        """
        # TODO: 在这里实现
        pass
    
    # ==================== 运算符重载 ====================
    
    def __add__(self, other):
        """
        加法运算: z = x + y
        
        TODO: 任务 1.2
        前向传播：
        - z.data = x.data + y.data
        
        反向传播（链式法则）：
        - dL/dx = dL/dz * dz/dx = dL/dz * 1
        - dL/dy = dL/dz * dz/dy = dL/dz * 1
        
        关键问题思考：
        1. 如果 x 和 y 的形状不同怎么办？（广播）
        2. 如果只有一个需要梯度怎么办？
        3. 梯度的形状应该是什么？
        
        提示：
        - 使用 numpy 的广播机制
        - 反向传播时需要处理广播的梯度求和
        """
        # TODO: 在这里实现
        pass
    
    def __mul__(self, other):
        """
        乘法运算: z = x * y（逐元素乘法）
        
        TODO: 任务 1.3
        前向传播：
        - z.data = x.data * y.data
        
        反向传播：
        - dL/dx = dL/dz * dz/dx = dL/dz * y
        - dL/dy = dL/dz * dz/dy = dL/dz * x
        
        关键问题思考：
        1. 为什么乘法的梯度是对方的值？
        2. 如何处理广播？
        """
        # TODO: 在这里实现
        pass
    
    def __sub__(self, other):
        """
        减法运算: z = x - y
        
        TODO: 实现减法（可以利用加法和负数）
        提示：x - y = x + (-1) * y
        """
        # TODO: 在这里实现
        pass
    
    def __truediv__(self, other):
        """
        除法运算: z = x / y
        
        TODO: 实现除法（可以利用乘法和倒数）
        提示：x / y = x * (1/y)
        """
        # TODO: 在这里实现
        pass
    
    def __pow__(self, power):
        """
        幂运算: z = x^n
        
        TODO: 实现幂运算
        反向传播：dz/dx = n * x^(n-1)
        """
        # TODO: 在这里实现
        pass
    
    def matmul(self, other):
        """
        矩阵乘法: Z = X @ Y
        
        TODO: 任务 1.4
        前向传播：
        - Z.data = X.data @ Y.data
        
        反向传播（重要！）：
        假设 Z = X @ Y
        - dL/dX = dL/dZ @ Y^T
        - dL/dY = X^T @ dL/dZ
        
        关键问题思考：
        1. 为什么是这样的梯度公式？（从维度推导）
        2. 批量矩阵乘法怎么处理？
        3. 如果 X 是 [m, n]，Y 是 [n, p]，梯度的形状是？
        
        形状分析：
        X: [m, n]
        Y: [n, p]
        Z: [m, p]
        dL/dZ: [m, p]
        dL/dX: [m, n] = [m, p] @ [p, n] = dL/dZ @ Y^T
        dL/dY: [n, p] = [n, m] @ [m, p] = X^T @ dL/dZ
        """
        # TODO: 在这里实现
        pass
    
    def sum(self, axis=None, keepdims=False):
        """
        求和运算
        
        TODO: 实现求和及其梯度
        反向传播：梯度广播到原始形状
        """
        # TODO: 在这里实现
        pass
    
    def mean(self, axis=None, keepdims=False):
        """
        求平均值
        
        TODO: 实现平均值及其梯度
        提示：mean = sum / count
        """
        # TODO: 在这里实现
        pass
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self):
        """打印 Tensor 信息"""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self):
        """返回形状"""
        return self.data.shape
    
    def numpy(self):
        """转换为 numpy 数组"""
        return self.data


# ==================== 测试代码 ====================

def test_basic_operations():
    """测试基本运算"""
    print("=" * 50)
    print("测试 1：基本运算")
    print("=" * 50)
    
    # 测试加法
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x + y
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x + y = {z.data}")
    
    # 反向传播
    z.backward()
    print(f"x.grad = {x.grad}")  # 应该是 [1, 1, 1]
    print(f"y.grad = {y.grad}")  # 应该是 [1, 1, 1]
    print()


def test_chain_rule():
    """测试链式法则"""
    print("=" * 50)
    print("测试 2：链式法则")
    print("=" * 50)
    
    # z = x * y + x
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    z = x * y + x  # z = 2*3 + 2 = 8
    
    print(f"z = x * y + x = {z.data}")
    
    z.backward()
    
    print(f"x.grad = {x.grad}")  # 应该是 y + 1 = 4
    print(f"y.grad = {y.grad}")  # 应该是 x = 2
    print()


def test_matmul():
    """测试矩阵乘法"""
    print("=" * 50)
    print("测试 3：矩阵乘法")
    print("=" * 50)
    
    # Z = X @ Y
    X = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # [2, 2]
    Y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)  # [2, 2]
    
    Z = X.matmul(Y)  # [2, 2]
    
    print(f"X =\n{X.data}")
    print(f"Y =\n{Y.data}")
    print(f"Z = X @ Y =\n{Z.data}")
    
    # 计算梯度
    Z.backward()
    
    print(f"X.grad =\n{X.grad}")
    print(f"Y.grad =\n{Y.grad}")
    print()


def test_gradient_check():
    """梯度检查（数值梯度 vs 自动微分梯度）"""
    print("=" * 50)
    print("测试 4：梯度检查")
    print("=" * 50)
    
    # 测试函数：f(x) = x^2
    x = Tensor([3.0], requires_grad=True)
    y = x * x
    
    # 自动微分梯度
    y.backward()
    auto_grad = x.grad
    
    # 数值梯度（有限差分）
    epsilon = 1e-5
    x_plus = Tensor([3.0 + epsilon])
    x_minus = Tensor([3.0 - epsilon])
    y_plus = x_plus * x_plus
    y_minus = x_minus * x_minus
    numerical_grad = (y_plus.data - y_minus.data) / (2 * epsilon)
    
    print(f"f(x) = x^2, x = 3.0")
    print(f"理论梯度 = 2x = 6.0")
    print(f"自动微分梯度 = {auto_grad}")
    print(f"数值梯度 = {numerical_grad}")
    print(f"误差 = {abs(auto_grad - numerical_grad)}")
    
    if abs(auto_grad - numerical_grad) < 1e-5:
        print("✅ 梯度检查通过！")
    else:
        print("❌ 梯度检查失败！")
    print()


if __name__ == "__main__":
    print("\n")
    print("🐻 Bears 学习之旅 - 阶段 1：自动微分")
    print("\n")
    
    # TODO: 完成 Tensor 类的实现后，运行以下测试
    # test_basic_operations()
    # test_chain_rule()
    # test_matmul()
    # test_gradient_check()
    
    print("\n")
    print("💡 提示：")
    print("1. 先实现 __init__ 和基本运算")
    print("2. 再实现 backward 方法")
    print("3. 使用梯度检查验证你的实现")
    print("4. 所有测试通过后，进入下一阶段")
    print("\n")
