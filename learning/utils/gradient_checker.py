"""
梯度检查工具
===================

用于验证你的自动微分实现是否正确
"""

import numpy as np


def numerical_gradient(func, x, epsilon=1e-5):
    """
    计算数值梯度（使用有限差分法）
    
    原理：
    f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)
    
    这是中心差分法，比单边差分更精确。
    
    Args:
        func: 函数 f(x)
        x: 输入点
        epsilon: 扰动量（推荐 1e-5）
    
    Returns:
        numerical_grad: 数值梯度
    """
    grad = np.zeros_like(x)
    
    # 对每个元素计算偏导数
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        
        # 保存原值
        old_value = x[idx]
        
        # f(x + epsilon)
        x[idx] = old_value + epsilon
        fx_plus = func(x)
        
        # f(x - epsilon)
        x[idx] = old_value - epsilon
        fx_minus = func(x)
        
        # 计算梯度
        grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        # 恢复原值
        x[idx] = old_value
        it.iternext()
    
    return grad


def check_gradient(func, x, auto_grad, epsilon=1e-5, threshold=1e-5):
    """
    检查自动微分梯度是否正确
    
    Args:
        func: 函数 f(x)
        x: 输入点（numpy array）
        auto_grad: 自动微分计算的梯度
        epsilon: 数值梯度的扰动量
        threshold: 误差阈值
    
    Returns:
        bool: 是否通过检查
    """
    # 计算数值梯度
    num_grad = numerical_gradient(func, x, epsilon)
    
    # 计算相对误差
    numerator = np.abs(auto_grad - num_grad)
    denominator = np.maximum(np.abs(auto_grad), np.abs(num_grad))
    relative_error = numerator / (denominator + 1e-8)
    
    max_error = np.max(relative_error)
    
    print(f"梯度检查:")
    print(f"  数值梯度: {num_grad}")
    print(f"  自动微分梯度: {auto_grad}")
    print(f"  最大相对误差: {max_error:.2e}")
    
    if max_error < threshold:
        print(f"  ✅ 通过！（误差 < {threshold}）")
        return True
    else:
        print(f"  ❌ 失败！（误差 >= {threshold}）")
        return False


# ==================== 使用示例 ====================

def example_usage():
    """梯度检查使用示例"""
    
    # 假设你实现了一个 Tensor 类
    # 这里用简单的函数演示
    
    def f(x):
        """测试函数: f(x) = x^2"""
        return np.sum(x ** 2)
    
    # 测试点
    x = np.array([1.0, 2.0, 3.0])
    
    # 计算数值梯度
    num_grad = numerical_gradient(f, x)
    print(f"数值梯度: {num_grad}")
    print(f"理论梯度: {2 * x}")
    
    # 检查
    check_gradient(f, x, auto_grad=2*x)


if __name__ == "__main__":
    example_usage()
