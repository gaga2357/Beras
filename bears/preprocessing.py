"""
Preprocessing 模块 - 实现数据预处理功能
"""

import numpy as np
import gzip
import struct
import os
import urllib.request


def download_mnist(path='./data'):
    """
    下载 MNIST 数据集
    
    Args:
        path: 保存路径
    """
    # 使用多个镜像源
    base_urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
    ]
    
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    # 创建目录
    os.makedirs(path, exist_ok=True)
    
    for filename in files:
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            
            # 尝试多个镜像源
            downloaded = False
            for base_url in base_urls:
                try:
                    url = base_url + filename
                    urllib.request.urlretrieve(url, filepath)
                    print(f'Downloaded {filename} from {base_url}')
                    downloaded = True
                    break
                except Exception as e:
                    print(f'Failed to download from {base_url}: {e}')
                    continue
            
            if not downloaded:
                raise Exception(f'Failed to download {filename} from all sources')
        else:
            print(f'{filename} already exists')


def load_mnist(path='./data', kind='train', download=True):
    """
    加载 MNIST 数据集
    
    Args:
        path: 数据集路径
        kind: 'train' 或 't10k'
        download: 如果文件不存在，是否自动下载
    
    Returns:
        images: 图像数据 [n_samples, 28, 28]
        labels: 标签数据 [n_samples]
    """
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    # 如果文件不存在且允许下载，则下载
    if download and (not os.path.exists(labels_path) or not os.path.exists(images_path)):
        download_mnist(path)
    
    # 加载标签
    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    # 加载图像
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows, cols)
    
    return images, labels


def load_mnist_simple(path='./data', download=True):
    """
    简化的 MNIST 加载函数，自动下载并加载数据
    
    Args:
        path: 数据集路径
        download: 是否自动下载（如果文件不存在）
    
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    try:
        # 尝试加载 gzip 格式
        X_train, y_train = load_mnist(path, kind='train', download=download)
        X_test, y_test = load_mnist(path, kind='t10k', download=download)
    except FileNotFoundError:
        # 如果没有 gzip 文件，尝试加载 numpy 文件
        try:
            data = np.load(os.path.join(path, 'mnist.npz'))
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
        except FileNotFoundError:
            raise FileNotFoundError(
                f"MNIST dataset not found in {path}. "
                "Please download MNIST dataset or provide correct path."
            )
    
    return (X_train, y_train), (X_test, y_test)


def create_dummy_mnist(n_train=1000, n_test=200):
    """
    创建虚拟 MNIST 数据集(用于测试)
    
    Args:
        n_train: 训练样本数
        n_test: 测试样本数
    
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    # 生成随机图像
    X_train = np.random.rand(n_train, 28, 28).astype(np.float32)
    X_test = np.random.rand(n_test, 28, 28).astype(np.float32)
    
    # 生成随机标签
    y_train = np.random.randint(0, 10, n_train)
    y_test = np.random.randint(0, 10, n_test)
    
    return (X_train, y_train), (X_test, y_test)


def normalize(X, method='minmax'):
    """
    归一化数据
    
    Args:
        X: 输入数据
        method: 归一化方法 ('minmax' 或 'standard')
    
    Returns:
        归一化后的数据
    """
    X = X.astype(np.float32)
    
    if method == 'minmax':
        # Min-Max 归一化: (x - min) / (max - min)
        X_min = X.min()
        X_max = X.max()
        X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    elif method == 'standard':
        # 标准化: (x - mean) / std
        X_mean = X.mean()
        X_std = X.std()
        X_normalized = (X - X_mean) / (X_std + 1e-8)
    elif method == 'scale':
        # 简单缩放: x / 255
        X_normalized = X / 255.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_normalized


def one_hot_encode(y, num_classes=10):
    """
    One-hot 编码
    
    Args:
        y: 标签数组 [n_samples]
        num_classes: 类别数量
    
    Returns:
        One-hot 编码后的数组 [n_samples, num_classes]
    """
    n_samples = len(y)
    y_one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    y_one_hot[np.arange(n_samples), y] = 1
    return y_one_hot


def flatten(X):
    """
    展平图像
    
    Args:
        X: 图像数据 [n_samples, height, width] 或 [n_samples, height, width, channels]
    
    Returns:
        展平后的数据 [n_samples, height * width * channels]
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def get_batches(X, y, batch_size=32, shuffle=True):
    """
    生成 mini-batch
    
    Args:
        X: 输入数据 [n_samples, ...]
        y: 标签数据 [n_samples, ...]
        batch_size: 批大小
        shuffle: 是否打乱数据
    
    Yields:
        (batch_X, batch_y): 批数据
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    划分训练集和测试集
    
    Args:
        X: 输入数据
        y: 标签数据
        test_size: 测试集比例
        shuffle: 是否打乱数据
        random_state: 随机种子
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def augment_images(X, rotation_range=15, shift_range=0.1, flip=False):
    """
    图像数据增强(简单实现)
    
    Args:
        X: 图像数据 [n_samples, height, width]
        rotation_range: 旋转角度范围(度)
        shift_range: 平移范围(比例)
        flip: 是否水平翻转
    
    Returns:
        增强后的图像数据
    """
    n_samples, height, width = X.shape
    X_augmented = np.copy(X)
    
    for i in range(n_samples):
        img = X[i]
        
        # 随机平移
        if shift_range > 0:
            shift_h = int(height * shift_range * (np.random.rand() - 0.5) * 2)
            shift_w = int(width * shift_range * (np.random.rand() - 0.5) * 2)
            img = np.roll(img, shift_h, axis=0)
            img = np.roll(img, shift_w, axis=1)
        
        # 随机翻转
        if flip and np.random.rand() > 0.5:
            img = np.fliplr(img)
        
        X_augmented[i] = img
    
    return X_augmented


def standardize(X, mean=None, std=None):
    """
    标准化数据
    
    Args:
        X: 输入数据
        mean: 均值(如果为 None，则从数据计算)
        std: 标准差(如果为 None，则从数据计算)
    
    Returns:
        标准化后的数据, 均值, 标准差
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    
    X_standardized = (X - mean) / (std + 1e-8)
    
    return X_standardized, mean, std


def to_categorical(y, num_classes=None):
    """
    将整数标签转换为分类矩阵(one-hot 编码)
    
    Args:
        y: 整数标签数组
        num_classes: 类别数量(如果为 None，则自动推断)
    
    Returns:
        One-hot 编码矩阵
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    return one_hot_encode(y, num_classes)


def pad_sequences(sequences, maxlen=None, padding='post', truncating='post', value=0.0):
    """
    填充序列到相同长度
    
    Args:
        sequences: 序列列表
        maxlen: 最大长度(如果为 None，则使用最长序列的长度)
        padding: 'pre' 或 'post'，在序列前或后填充
        truncating: 'pre' 或 'post'，从序列前或后截断
        value: 填充值
    
    Returns:
        填充后的数组
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    n_samples = len(sequences)
    padded = np.full((n_samples, maxlen), value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        seq = np.array(seq, dtype=np.float32)
        if len(seq) > maxlen:
            # 截断
            if truncating == 'pre':
                padded[i] = seq[-maxlen:]
            else:
                padded[i] = seq[:maxlen]
        else:
            # 填充
            if padding == 'pre':
                padded[i, -len(seq):] = seq
            else:
                padded[i, :len(seq)] = seq
    
    return padded
