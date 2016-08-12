from __future__ import print_function, absolute_import, division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
figsize(12, 4)

import os
import sys
os.environ['THEANO_FLAGS'] = "device=cpu,optimizer=fast_run"
DATA_DIR = os.path.join('/res', 'data')
sys.path.append(os.path.join('/res', 'src'))

import numpy as np

X = np.linspace(-10, 10, 1000)
y = X
plt.figure()
plt.plot(X, y)
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.linspace(-10, 10, 1000)
y = sigmoid(X)
plt.figure()
plt.plot(X, y)
plt.show()

X = np.linspace(-12, 12, 1000)
y = np.tanh(X)
plt.figure()
plt.plot(X, y)
plt.show()

X = np.linspace(-12, 12, 1000)
y = [max(i, 0) for i in X]
plt.figure()
plt.plot(X, y)
plt.show()

X = np.linspace(-12, 12, 1000)
y = np.where(X > 0, X, 0.01 * X)
plt.figure()
plt.plot(X, y)
plt.show()
