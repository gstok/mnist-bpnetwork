#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np;

class bpNet:
    # 构造函数
    def __init__ (
        self,
        inputSize =  785,
        outputSize = 10,
        hiddenLayersSize = [300, 200, 100, 50, 20],
    ):
        self.inputSize = inputSize;
        self.outputSize = outputSize;
        self.hiddenLayersSize = hiddenLayersSize;
        self.params = self.initParams();

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
        param["weight"] = np.random.randn(inputSize, outputSize);
        param["bias"] = np.zeros(outputSize);
        return param;
