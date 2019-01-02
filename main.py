#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet.bpNet import bpNet;

# 新建mnist数据集对象，初始化数据集
mstData = mnist();

# 提取mnist数据
trainImg = mstData.trainImg;
trainLabel = mstData.trainLabel;
testImg = mstData.testImg;
testLabel = mstData.testLabel;

# 训练数据大小
trainSize = trainImg.shape[0];
# 训练批大小
batchSize = 100;
# 迭代次数
itersNum = 10000;
# 学习率
learningRate = 0.1;

network = bpNet();

for index in range(itersNum):
    # 获取随机选取的索引
    choiceIndexs = np.random.choice(trainSize, batchSize);
    imgs = trainImg[choiceIndexs];
    labels = trainLabel[choiceIndexs];
    network.gradient(imgs, labels);
    print(network.lastLayer.loss);
    network.update();


# net = bpNet();
# y = net.gradient(mstData.trainImg, mstData.trainLabel);
# print(len(net.hiddenLayers));


