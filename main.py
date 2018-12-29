#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet.bpNet import bpNet;
from common.layers import reLu;

# net = bpNet();
a = reLu();
x = np.array([
    [1, -1, -0.1],
    [1, 1, 0],
]);
r = a.forward(x);
b = a.backward(np.array([
    [9, -1, 1],
    [-0.31, 2, 3],
]));
print(b);

