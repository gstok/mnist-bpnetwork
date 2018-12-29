#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os;
import collections;
import gzip;
import pickle;
import numpy as np;
try:
    import urllib.request;
except ImportError:
    raise ImportError("You should use Python 3.x");

# 当前模块路径
curDirPath = os.path.dirname(os.path.abspath(__file__)) + "/";
# mnist数据集基础下载地址
mnistBaseUrl = "http://yann.lecun.com/exdb/mnist/";
# mnist数据集下载文件信息Map
fileMap = collections.OrderedDict();
fileMap["trainImg"] = "train-images-idx3-ubyte.gz";
fileMap["trainLabel"] = "train-labels-idx1-ubyte.gz";
fileMap["testImg"] = "t10k-images-idx3-ubyte.gz";
fileMap["testLabel"] = "t10k-labels-idx1-ubyte.gz";

class mnist:
    # 构造函数
    def __init__ (self, normalize = True, flatten = True, oneHotLabel = True):
        self.pklPath = curDirPath + "mnist.pkl";
        self.initFileMap();
        if os.path.exists(self.pklPath):
            self.loadPKLObject(self.pklPath);
        else:
            self.tryToDownloadMnist();
            self.loadMnist();
        if (normalize):
            self.trainImg = self.imageNormalize(self.trainImg);
            self.testImg = self.imageNormalize(self.testImg);
        if (not flatten):
            self.trainImg = self.imageRestore(self.trainImg);
            self.testImg = self.imageRestore(self.testImg);
        if (oneHotLabel):
            self.trainLabel = self.changeOneHotLabel(self.trainLabel);
            self.testLabel = self.changeOneHotLabel(self.testLabel);
    # 整理mnist文件资源Map，整理出下载地址以及下载路径
    def initFileMap (self):
        self.fileMap = collections.OrderedDict();
        for key, value in fileMap.items():
            file, ext = os.path.splitext(value);
            self.fileMap[key] = {
                "downUrl": mnistBaseUrl + value,
                "downPath": curDirPath + key + ext,
            };
    # 尝试下载mnist数据集
    def tryToDownloadMnist (self):
        for value in self.fileMap.values():
            downUrl = value["downUrl"];
            downPath = value["downPath"];
            if (os.path.exists(value["downPath"])):
                pass;
            else:
                print("尝试下载 %s ..." % downUrl);
                urllib.request.urlretrieve(downUrl, downPath);
    # 从文件中加载图像数据
    def loadImg (self, filePath):
        print("加载图像数据 %s ..." % filePath);
        with gzip.open(filePath) as f:
            fileBuf = f.read();
            data = np.frombuffer(fileBuf, np.uint8, offset = 16);
        data = data.reshape(-1, 784);
        return data;
    # 从文件中加载标签数据
    def loadLabel (self, filePath):
        print("加载标签数据 %s ..." % filePath);
        with gzip.open(filePath) as f:
            fileBuf = f.read();
            data = np.frombuffer(fileBuf, np.uint8, offset = 8);
        return data;
    # 从已下载的文件中加载mnist数据集
    def loadMnist (self):
        tmpMap = collections.OrderedDict();
        for key, value in self.fileMap.items():
            downPath = value["downPath"];
            if (key.lower().endswith("img")):
                tmpMap[key] = self.loadImg(downPath);
            elif (key.lower().endswith("label")):
                tmpMap[key] = self.loadLabel(downPath);
        with open(self.pklPath, "wb") as f:
            pickle.dump(tmpMap, f, -1);
        self.trainImg = tmpMap["trainImg"];
        self.trainLabel = tmpMap["trainLabel"];
        self.testImg = tmpMap["testImg"];
        self.testLabel = tmpMap["testLabel"];
    # 从pkl文件之中恢复对象
    def loadPKLObject (self, pklPath):
        obj = { };
        with open(pklPath, "rb") as f:
            obj = pickle.load(f);
        self.trainImg = obj["trainImg"];
        self.trainLabel = obj["trainLabel"];
        self.testImg = obj["testImg"];
        self.testLabel = obj["testLabel"];
    # 图像正规化
    def imageNormalize (self, data):
        newData = data.astype(np.float);
        return newData / 255.0;
    # 图像恢复形状
    def imageRestore (self, data):
        return data.reshape(-1, 1, 28, 28);
    # 修改标签为OneHot模式
    def changeOneHotLabel (self, data):
        newData = np.zeros((data.size, 10));
        for index, row in enumerate(newData):
            row[data[index]] = 1;
        return newData;
