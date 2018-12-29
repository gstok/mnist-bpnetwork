#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet.bpNet import bpNet;
from common.layers import reLu;

a = np.array([
    1,
    2,
]);
b = np.array([
    [1, 2, 3],
    [4, 5, 6],
]);
print(np.dot(b, a));

