#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os;
import collections;
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
    def __init__ (self):
        self.initFileMap();
        self.tryToDownloadMnist();
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
                print("%s 文件已经存在" % downPath);
            else:
                print("尝试下载 %s ..." % downUrl);
                urllib.request.urlretrieve(downUrl, downPath);