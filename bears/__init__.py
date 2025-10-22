"""
Bears 🐻 - 轻量级深度学习框架

一个教育性质的深度学习框架，模仿 Keras API，所有核心功能从底层实现。
"""

__version__ = '0.1.0'
__author__ = 'Bears Team'

from .tensor import Tensor, relu, sigmoid, softmax, log, exp
from .layers import Layer, Dense, ReLU, Sigmoid, Softmax
from .losses import Loss, MSELoss, CrossEntropyLoss
from .metrics import accuracy, mse_metric
from .optimizers import Optimizer, SGD, Adam
from .models import Sequential
from .preprocessing import (
    load_mnist, load_mnist_simple, create_dummy_mnist,
    normalize, one_hot_encode, flatten, get_batches,
    train_test_split, to_categorical
)

__all__ = [
    'Tensor', 'relu', 'sigmoid', 'softmax', 'log', 'exp',
    'Layer', 'Dense', 'ReLU', 'Sigmoid', 'Softmax',
    'Loss', 'MSELoss', 'CrossEntropyLoss',
    'accuracy', 'mse_metric',
    'Optimizer', 'SGD', 'Adam',
    'Sequential',
    'load_mnist', 'load_mnist_simple', 'create_dummy_mnist',
    'normalize', 'one_hot_encode', 'flatten', 'get_batches',
    'train_test_split', 'to_categorical'
]
