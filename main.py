#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os;
import os.path;
import numpy as np;
from mnist.mnist import mnist;
from bpNet import bpNet;

mstData = mnist();

net = bpNet();

