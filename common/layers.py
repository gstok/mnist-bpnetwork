
import numpy as np;

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