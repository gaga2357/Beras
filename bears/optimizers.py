"""
Optimizers 模块 - 实现优化器
"""

import numpy as np


class Optimizer:
    """
    优化器基类
    """
    
    def step(self, parameters):
        """
        更新参数
        
        Args:
            parameters: 参数列表
        """
        raise NotImplementedError("Subclass must implement step method")
    
    def zero_grad(self, parameters):
        """
        清零梯度
        
        Args:
            parameters: 参数列表
        """
        for param in parameters:
            if param.requires_grad:
                param.zero_grad()


class SGD(Optimizer):
    """
    随机梯度下降优化器 (Stochastic Gradient Descent)
    
    更新规则: θ = θ - η * ∇L(θ)
    
    支持动量(momentum)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        初始化 SGD 优化器
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数 (0 表示不使用动量)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}  # 存储动量
    
    def step(self, parameters):
        """
        更新参数
        
        Args:
            parameters: 参数列表
        """
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            if self.momentum > 0:
                # 使用动量
                if i not in self.velocities:
                    self.velocities[i] = np.zeros_like(param.data)
                
                # v = momentum * v - lr * grad
                self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * param.grad
                
                # θ = θ + v
                param.data += self.velocities[i]
            else:
                # 标准 SGD
                # θ = θ - lr * grad
                param.data -= self.learning_rate * param.grad
    
    def __repr__(self):
        return f"SGD(learning_rate={self.learning_rate}, momentum={self.momentum})"


class Adam(Optimizer):
    """
    Adam 优化器 (Adaptive Moment Estimation)
    
    自适应学习率优化器，结合了动量和 RMSprop
    
    更新规则:
        m = β1 * m + (1 - β1) * ∇L
        v = β2 * v + (1 - β2) * ∇L²
        m_hat = m / (1 - β1^t)
        v_hat = v / (1 - β2^t)
        θ = θ - η * m_hat / (√v_hat + ε)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        初始化 Adam 优化器
        
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            epsilon: 数值稳定性常数
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步
    
    def step(self, parameters):
        """
        更新参数
        
        Args:
            parameters: 参数列表
        """
        self.t += 1
        
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            # 初始化矩估计
            if i not in self.m:
                self.m[i] = np.zeros_like(param.data)
                self.v[i] = np.zeros_like(param.data)
            
            # 更新一阶矩估计
            # m = β1 * m + (1 - β1) * grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # 更新二阶矩估计
            # v = β2 * v + (1 - β2) * grad²
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            # θ = θ - lr * m_hat / (√v_hat + ε)
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def __repr__(self):
        return f"Adam(learning_rate={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2})"


class RMSprop(Optimizer):
    """
    RMSprop 优化器
    
    自适应学习率优化器
    
    更新规则:
        v = β * v + (1 - β) * ∇L²
        θ = θ - η * ∇L / (√v + ε)
    """
    
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        """
        初始化 RMSprop 优化器
        
        Args:
            learning_rate: 学习率
            beta: 指数衰减率
            epsilon: 数值稳定性常数
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        
        self.v = {}  # 二阶矩估计
    
    def step(self, parameters):
        """
        更新参数
        
        Args:
            parameters: 参数列表
        """
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            # 初始化矩估计
            if i not in self.v:
                self.v[i] = np.zeros_like(param.data)
            
            # 更新二阶矩估计
            # v = β * v + (1 - β) * grad²
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (param.grad ** 2)
            
            # 更新参数
            # θ = θ - lr * grad / (√v + ε)
            param.data -= self.learning_rate * param.grad / (np.sqrt(self.v[i]) + self.epsilon)
    
    def __repr__(self):
        return f"RMSprop(learning_rate={self.learning_rate}, beta={self.beta})"


class AdaGrad(Optimizer):
    """
    AdaGrad 优化器
    
    自适应学习率优化器，对不同参数使用不同的学习率
    
    更新规则:
        v = v + ∇L²
        θ = θ - η * ∇L / (√v + ε)
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        初始化 AdaGrad 优化器
        
        Args:
            learning_rate: 学习率
            epsilon: 数值稳定性常数
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.v = {}  # 梯度平方累积
    
    def step(self, parameters):
        """
        更新参数
        
        Args:
            parameters: 参数列表
        """
        for i, param in enumerate(parameters):
            if not param.requires_grad or param.grad is None:
                continue
            
            # 初始化累积
            if i not in self.v:
                self.v[i] = np.zeros_like(param.data)
            
            # 累积梯度平方
            # v = v + grad²
            self.v[i] += param.grad ** 2
            
            # 更新参数
            # θ = θ - lr * grad / (√v + ε)
            param.data -= self.learning_rate * param.grad / (np.sqrt(self.v[i]) + self.epsilon)
    
    def __repr__(self):
        return f"AdaGrad(learning_rate={self.learning_rate})"
