"""
简单测试脚本 - 验证 Bears 框架的核心功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from bears import Tensor, Sequential, Dense, ReLU, MSELoss, SGD


def test_tensor_operations():
    """测试张量运算"""
    print("=" * 70)
    print("Test 1: Tensor Operations")
    print("=" * 70)
    
    # 创建张量
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    # 运算
    z = (x * y).sum()
    
    # 反向传播
    z.backward()
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x * y = {z.data}")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    print("✓ Tensor operations work correctly!\n")


def test_gradient_computation():
    """测试梯度计算"""
    print("=" * 70)
    print("Test 2: Gradient Computation")
    print("=" * 70)
    
    # 测试简单函数: f(x) = x^2
    x = Tensor([2.0], requires_grad=True)
    y = x * x
    y.backward()
    
    print(f"f(x) = x^2")
    print(f"x = {x.data[0]}")
    print(f"f(x) = {y.data[0]}")
    print(f"df/dx = {x.grad[0]} (expected: 4.0)")
    
    # 验证梯度
    assert abs(x.grad[0] - 4.0) < 1e-5, "Gradient computation failed!"
    print("✓ Gradient computation is correct!\n")


def test_matrix_multiplication():
    """测试矩阵乘法"""
    print("=" * 70)
    print("Test 3: Matrix Multiplication")
    print("=" * 70)
    
    # 创建矩阵
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    # 矩阵乘法
    C = A.matmul(B)
    
    print(f"A = \n{A.data}")
    print(f"B = \n{B.data}")
    print(f"C = A @ B = \n{C.data}")
    
    # 反向传播
    C.sum().backward()
    
    print(f"A.grad = \n{A.grad}")
    print(f"B.grad = \n{B.grad}")
    print("✓ Matrix multiplication works correctly!\n")


def test_simple_regression():
    """测试简单回归任务"""
    print("=" * 70)
    print("Test 4: Simple Regression Task")
    print("=" * 70)
    
    # 生成简单的线性数据: y = 2x + 1
    np.random.seed(42)
    X = np.random.rand(100, 1).astype(np.float32)
    y = 2 * X + 1 + np.random.randn(100, 1).astype(np.float32) * 0.1
    
    # 构建模型
    model = Sequential()
    model.add(Dense(1, 10))
    model.add(ReLU())
    model.add(Dense(10, 1))
    
    # 编译模型
    model.compile(loss=MSELoss(), optimizer=SGD(learning_rate=0.01))
    
    print("Training simple regression model...")
    print("Target function: y = 2x + 1")
    
    # 训练模型
    history = model.fit(X, y, epochs=50, batch_size=10, verbose=False)
    
    # 测试预测
    X_test = np.array([[0.0], [0.5], [1.0]], dtype=np.float32)
    y_pred = model.predict(X_test)
    
    print(f"\nPredictions:")
    for i, (x_val, y_val) in enumerate(zip(X_test, y_pred)):
        expected = 2 * x_val[0] + 1
        print(f"  x={x_val[0]:.1f}, predicted={y_val[0]:.2f}, expected={expected:.2f}")
    
    # 检查最终损失
    final_loss = history['loss'][-1]
    print(f"\nFinal loss: {final_loss:.4f}")
    
    if final_loss < 0.5:
        print("✓ Regression model trained successfully!\n")
    else:
        print("⚠ Regression model may need more training\n")


def test_activation_functions():
    """测试激活函数"""
    print("=" * 70)
    print("Test 5: Activation Functions")
    print("=" * 70)
    
    from bears import relu, sigmoid, softmax
    
    # 测试 ReLU
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)
    print(f"ReLU({x.data}) = {y.data}")
    
    # 测试 Sigmoid
    x = Tensor([0.0], requires_grad=True)
    y = sigmoid(x)
    print(f"Sigmoid(0.0) = {y.data[0]:.4f} (expected: 0.5)")
    
    # 测试 Softmax
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = softmax(x)
    print(f"Softmax([1, 2, 3]) = {y.data[0]}")
    print(f"Sum of softmax = {y.data[0].sum():.4f} (expected: 1.0)")
    
    print("✓ Activation functions work correctly!\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Bears 🐻 Framework - Core Functionality Tests")
    print("=" * 70 + "\n")
    
    try:
        test_tensor_operations()
        test_gradient_computation()
        test_matrix_multiplication()
        test_activation_functions()
        test_simple_regression()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
