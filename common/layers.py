
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

        # 不知这是做什么的
        self.original_x_shape = None;
        
    def forward (self, x):

        # 不知这是做什么的
        self.original_x_shape = x.shape;
        x = x.reshape(x.shape[0], -1);

        self.x = x;
        return np.dot(x, self.weight) + self.bias;
    def backward (self, d):
        self.xD = np.dot(d, self.weight.T);
        self.weightD = np.dot(self.x.T, d);
        self.biasD = np.sum(d, axis = 0);

        # 不知这是做什么的
        self.xD = self.xD.reshape(* self.original_x_shape);

        return self.xD;

