#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet.bpNet import bpNet;
import pickle;

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
itersNum = 100000;
# 学习率
learningRate = 0.1;
# 轮数
iterPerEpoch = max((trainSize / batchSize), 1);

network = None;

# 利用网络模型求导
# with open("network.pkl", "rb") as f:
#     network = pickle.load(f);

# los = network.predict(testImg[109]);
# print(np.argmax(los));
# print(testLabel[109]);

network = bpNet();

for index in range(itersNum):
    if (index % iterPerEpoch == 0):
        trainAcc = network.accuracy(trainImg, trainLabel);
        testAcc = network.accuracy(testImg, testLabel);
        print("训练精度: %s  测试精度: %s" % (trainAcc, testAcc));
    # 获取随机选取的索引
    trainIndexs = np.random.choice(trainSize, batchSize);
    imgs = trainImg[trainIndexs];
    labels = trainLabel[trainIndexs];
    network.update(imgs, labels);

with open("network.pkl", "wb") as f:
    pickle.dump(network, f, -1);



