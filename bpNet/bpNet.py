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
        hiddenLayersSize = [300, 200, 100, 50, 20],
    ):
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        self.hiddenLayersSize = hiddenLayersSize;
        self.params = self.initParams();
        self.hiddenLayers = self.initHiddenLayers();

    # 使用神经网络进行预测
    def predict (self, x):
        y = x.copy();
        for layer in self.hiddenLayers:
            y = layer.forward(y);
        return y;


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
        param["weight"] = np.random.randn(inputSize, outputSize);
        param["bias"] = np.zeros(outputSize);
        return param;
