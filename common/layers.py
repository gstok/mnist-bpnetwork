
import numpy as np;
from common.functions import *;

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

        # 不知这是做什么的
        self.original_x_shape = None;
        
    def forward (self, x):

        # 不知这是做什么的
        # self.original_x_shape = x.shape;
        # x = x.reshape(x.shape[0], -1);

        self.x = x;
        return np.dot(x, self.weight) + self.bias;
    def backward (self, d):
        self.xD = np.dot(d, self.weight.T);
        self.weightD = np.dot(self.x.T, d);
        self.biasD = np.sum(d, axis = 0);

        # 不知这是做什么的
        self.xD = self.xD.reshape(* self.original_x_shape);

        return self.xD;

# 仿射ReLu层
class affineReLu:
    def __init__ (self, weight, bias):
        self.affineLayer = affine(weight, bias);
        self.reLuLayer = reLu();
    def forward (self, x):
        y = self.affineLayer.forward(x);
        y = self.reLuLayer.forward(y);
        return y;
    def backward (self, d):
        outD = self.reLuLayer.backward(d);
        outD = self.affineLayer.backward(outD);
        return outD;

class softmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = crossEntropyError(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx



