
import numpy as np;

# relu层实现
class reLu:
    def __init__ (self):
        self.mask = None;
    # 前向传播（计算结果）
    def forward (self, x):
        self.mask = (x <= 0);
        # numpy数组拷贝
        result = x.copy();
        result[self.mask] = 0;
        return result;
    # 反向传播（求导）
    def backward (self, d):
        result = d.copy();
        result[self.mask] = 0;
        return result;

# sigmoid层实现
class sigmoid:
    def __init__ (self):
        pass;
    def forward (self, x):
        self.y = 1.0 / (1.0 + np.exp(-x));
        return self.y;
    def backward (self, d):
        return d * self.y * (1.0 - self.y);

# affine层的实现
class affine:
    def __init__ (self, weight, bias):
        self.x = None;
        self.weight = weight;
        self.bias = bias;
        self.xD = None;
        self.weightD = None;
        self.biasD = None;
    def forward (self, x):
        self.x = x;
        return np.dot(x, self.weight) + self.bias;
    def backward (self, d):
        self.xD = np.dot(d, self.weight.T);
        self.weightD = np.dot(d, self.x.T);
        self.biasD = np.sum(d, axis = 0);
        return self.xD;

