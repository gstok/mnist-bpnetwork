
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