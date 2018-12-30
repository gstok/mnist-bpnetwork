#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet.bpNet import bpNet;

mstData = mnist();
img = mstData.trainImg[100];
label = mstData.trainLabel[100];
net = bpNet();
net.predict(mstData.trainImg);