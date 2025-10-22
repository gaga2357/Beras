"""
Models 模块 - 实现模型封装
"""

import numpy as np
from .tensor import Tensor
from .layers import Layer


class Sequential:
    """
    顺序模型
    
    按顺序堆叠层，提供训练和预测接口
    
    类似于 Keras 的 Sequential 模型
    """
    
    def __init__(self, layers=None):
        """
        初始化顺序模型
        
        Args:
            layers: 层列表(可选)
        """
        self.layers = layers if layers is not None else []
        self.loss_fn = None
        self.optimizer = None
        self.compiled = False
    
    def add(self, layer):
        """
        添加层
        
        Args:
            layer: 要添加的层
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected Layer instance, got {type(layer)}")
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """
        配置模型训练
        
        Args:
            loss: 损失函数
            optimizer: 优化器
        """
        self.loss_fn = loss
        self.optimizer = optimizer
        self.compiled = True
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入数据（numpy array 或 Tensor）
        
        Returns:
            输出张量
        """
        # 如果输入是 numpy array，转换为 Tensor
        if isinstance(x, np.ndarray):
            from .tensor import Tensor
            x = Tensor(x, requires_grad=False)
        
        for layer in self.layers:
            x = layer(x)
        return x
    
    def __call__(self, x):
        """调用接口"""
        return self.forward(x)
    
    def get_parameters(self):
        """
        获取所有可训练参数
        
        Returns:
            参数列表
        """
        parameters = []
        for layer in self.layers:
            if layer.trainable:
                parameters.extend(layer.get_parameters())
        return parameters
    
    def fit(self, X, y, epochs=10, batch_size=32, verbose=True, validation_data=None):
        """
        训练模型
        
        Args:
            X: 训练数据 [n_samples, n_features]
            y: 训练标签 [n_samples, n_outputs]
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印训练信息
            validation_data: 验证数据 (X_val, y_val)
        
        Returns:
            训练历史 (字典)
        """
        if not self.compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile() first.")
        
        # 转换为 numpy array
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        if isinstance(y, list):
            y = np.array(y, dtype=np.float32)
        
        n_samples = len(X)
        history = {
            'loss': [],
            'val_loss': [] if validation_data is not None else None
        }
        
        for epoch in range(epochs):
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch 训练
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_X = X_shuffled[i:batch_end]
                batch_y = y_shuffled[i:batch_end]
                
                # 转换为 Tensor
                x_tensor = Tensor(batch_X, requires_grad=False)
                y_tensor = Tensor(batch_y, requires_grad=False)
                
                # 前向传播
                y_pred = self.forward(x_tensor)
                
                # 计算损失
                loss = self.loss_fn(y_pred, y_tensor)
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                parameters = self.get_parameters()
                self.optimizer.step(parameters)
                
                # 清零梯度
                self.optimizer.zero_grad(parameters)
                
                # 记录损失
                epoch_loss += loss.data
                n_batches += 1
            
            # 平均损失
            avg_loss = epoch_loss / n_batches
            history['loss'].append(float(avg_loss))
            
            # 验证
            val_loss_str = ""
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.evaluate(X_val, y_val, return_loss=True)
                history['val_loss'].append(val_loss)
                val_loss_str = f", val_loss: {val_loss:.4f}"
            
            # 打印信息
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}{val_loss_str}")
        
        return history
    
    def predict(self, X):
        """
        预测
        
        Args:
            X: 输入数据 [n_samples, n_features]
        
        Returns:
            预测结果 (numpy array)
        """
        # 转换为 numpy array
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        
        # 转换为 Tensor
        x_tensor = Tensor(X, requires_grad=False)
        
        # 前向传播
        y_pred = self.forward(x_tensor)
        
        return y_pred.data
    
    def evaluate(self, X, y, return_loss=False):
        """
        评估模型
        
        Args:
            X: 测试数据 [n_samples, n_features]
            y: 测试标签 [n_samples, n_outputs]
            return_loss: 是否返回损失值
        
        Returns:
            如果 return_loss=True，返回损失值；否则返回预测结果
        """
        # 转换为 numpy array
        if isinstance(X, list):
            X = np.array(X, dtype=np.float32)
        if isinstance(y, list):
            y = np.array(y, dtype=np.float32)
        
        # 预测
        y_pred = self.predict(X)
        
        if return_loss:
            # 计算损失
            y_pred_tensor = Tensor(y_pred, requires_grad=False)
            y_tensor = Tensor(y, requires_grad=False)
            loss = self.loss_fn(y_pred_tensor, y_tensor)
            return float(loss.data)
        else:
            return y_pred
    
    def summary(self):
        """
        打印模型摘要
        """
        print("=" * 70)
        print("Model: Sequential")
        print("=" * 70)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 70)
        
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{i}"
            
            # 计算参数数量
            params = layer.get_parameters()
            n_params = sum(p.data.size for p in params)
            total_params += n_params
            
            # 输出形状(简化显示)
            if hasattr(layer, 'output_dim'):
                output_shape = f"(None, {layer.output_dim})"
            else:
                output_shape = "N/A"
            
            print(f"{layer_name:<30} {output_shape:<20} {n_params:<10}")
        
        print("=" * 70)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {total_params}")
        print(f"Non-trainable params: 0")
        print("=" * 70)
    
    def save_weights(self, filepath):
        """
        保存模型权重
        
        Args:
            filepath: 保存路径
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            if layer.trainable:
                params = layer.get_parameters()
                weights[f'layer_{i}'] = [p.data for p in params]
        
        np.save(filepath, weights)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """
        加载模型权重
        
        Args:
            filepath: 加载路径
        """
        weights = np.load(filepath, allow_pickle=True).item()
        
        for i, layer in enumerate(self.layers):
            if layer.trainable and f'layer_{i}' in weights:
                params = layer.get_parameters()
                layer_weights = weights[f'layer_{i}']
                for param, weight in zip(params, layer_weights):
                    param.data = weight
        
        print(f"Model weights loaded from {filepath}")
    
    def __repr__(self):
        layers_str = "\n  ".join([str(layer) for layer in self.layers])
        return f"Sequential(\n  {layers_str}\n)"
