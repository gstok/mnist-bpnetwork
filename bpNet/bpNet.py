#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np;
from common.layers import *;

class bpNet:
    # 构造函数
    def __init__ (
        self,
        inputSize =  784,
        outputSize = 10,
        hiddenLayersSize = [100, 50],
        weightInitStd = 0.01,
    ):
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        self.hiddenLayersSize = hiddenLayersSize;
        self.weightInitStd = weightInitStd;
        self.params = self.initParams();
        self.hiddenLayers = self.initHiddenLayers();
        self.lastLayer = softmaxLoss();

    # 使用神经网络进行预测
    def predict (self, x):
        y = x.copy();
        for layer in self.hiddenLayers:
            y = layer.forward(y);
        return y;

    # 计算损失函数
    def loss (self, x, t):
        y = self.predict(x);
        return self.lastLayer.forward(y, t);

    # 反向传播计算梯度
    def gradient (self, x, t):
        self.loss(x, t);
        d = 1;
        d = self.lastLayer.backward(d);
        for index in range(len(self.hiddenLayers) - 1, -1, -1):
            d = self.hiddenLayers[index].backward(d);

    # 计算网络预测精度
    def accuracy (self):
        a = np.argmax(self.lastLayer.y, axis = 1);
        b = np.argmax(self.lastLayer.t, axis = 1);
        c = a != b;
        print(np.sum(c));

    # 根据保存在各层的梯度更新神经网络参数
    def update (self, learningRate = 0.3):
        for layer in self.hiddenLayers:
            weight = None;
            bias = None;
            if (isinstance(layer, affineReLu)):
                layer.affineLayer.weight -= layer.affineLayer.weightD * learningRate;
                layer.affineLayer.bias -= layer.affineLayer.bias * learningRate;
            elif (isinstance(layer, affine)):
                layer.weight -= layer.weightD * learningRate;
                layer.bias -= layer.biasD * learningRate;

    # 根据初始化的参数构建隐藏层
    def initHiddenLayers (self):
        layers = [];
        for index, value in enumerate(self.params):
            weight = value["weight"];
            bias = value["bias"];
            layer = None;
            if (index == len(self.params) - 1):
                layer = affine(weight, bias);
            else:
                layer = affineReLu(weight, bias);
            layers.append(layer);
        return layers;

    # 初始化各层参数
    def initParams (self):
        params = [];
        layerSizeList = [ self.inputSize ] + self.hiddenLayersSize + [ self.outputSize ];
        for index, value in enumerate(layerSizeList):
            if (index > 0):
                prevSize = layerSizeList[index - 1];
                curSize = value;
                param = self.initLayerParam(prevSize, curSize);
                params.append(param);
        return params;
    # 初始化层参数，包括权重和偏置
    def initLayerParam (self, inputSize, outputSize):
        param = { };
        # 利用高斯分布初始化权重矩阵
        param["weight"] = self.weightInitStd * np.random.randn(inputSize, outputSize);
        param["bias"] = self.weightInitStd * np.zeros(outputSize);
        return param;
