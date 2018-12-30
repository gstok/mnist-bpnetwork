
import numpy as np;

# 旧的softmax函数，仅仅支持单个维度，此函数的作用是取0~1的概率分布
def softmax1 (x):
    c = np.max(x);
    expA = np.exp(x - c);
    sumExpA = np.sum(expA);
    y = expA / sumExpA;
    return y;

# 支持批处理的softmax函数
def softmax (x):
    cpx = x.copy();
    if (cpx.ndim == 2):
        cpx = cpx.T;
        cpx -= np.max(cpx, axis = 0);
        y = np.exp(cpx) / np.sum(np.exp(cpx), axis = 0);
        return y.T;
    else:
        cpx -= np.max(cpx);
        return np.exp(cpx) / np.sum(np.exp(cpx));

def crossEntropyError(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size