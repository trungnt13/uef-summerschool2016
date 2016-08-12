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

import scipy.io as sio
import numpy as np

import theano
from theano import tensor as T

import h5py # for loading data

from utils import (plot_images, Progbar, plot_confusion_matrix, plot_weights,
                   glorot_uniform, validate_deep_network, train_networks,
                   report_performance, plot_weights4D)

f = h5py.File(os.path.join(DATA_DIR, 'mnist.h5'), 'r')
X_train = f['X_train'].value
y_train = f['y_train'].value

X_valid = f['X_valid'].value
y_valid = f['y_valid'].value

X_test = f['X_test'].value
y_test = f['y_test'].value

nb_channel = 1
nb_new_channel = 8
filter_size = (3, 3)
nb_features = (28, 28)
nb_classes = 10

# Original data is set of digits images (nb_samples, width, height)
# We have grey scale images, so nb_channel equal to 1
# Hence, we have to reshape data to  (nb_samples, nb_channel, width, height)
X = T.tensor3(name='X', dtype='float32')
# this is how we add 1 dimension add 'x' position
X_ = X.dimshuffle((0, 'x', 1, 2))
y_true = T.ivector(name='y')

W_conv = theano.shared(glorot_uniform(shape=(nb_new_channel, nb_channel,) + filter_size),
                       name='W_conv')
# the shape of bias must match the new shape of image after convolution
b_conv = theano.shared(np.zeros(shape=(nb_new_channel, 26, 26), dtype='float32'),
                  name='b_conv')

W_proj = theano.shared(glorot_uniform(shape=(nb_new_channel * 26 * 26, nb_classes)),
                       name='W_proj')
b_proj = theano.shared(np.zeros(shape=(nb_classes,), dtype='float32'),
                       name='b_proj')

# instead of dot product, we use convolution operator here
activation = T.nnet.conv2d(X_, filters=W_conv,
                           border_mode='valid',
                           subsample=(1, 1),
                           filter_flip=False)
# activate the output with relu
activation = T.nnet.relu(activation + b_conv)
# store conv output for later use
conv_output = activation

# convert to 2D so we can project it to output
activation = T.flatten(activation, outdim=2)
activation = T.nnet.softmax(T.dot(activation, W_proj) + b_proj)
y_pred = activation

cost = T.mean(T.nnet.categorical_crossentropy(y_pred, y_true))
learning_rate = theano.shared(np.cast['float32'](0.1), name='learning_rate')
grad = T.grad(cost, wrt=[W_conv, b_conv, W_proj, b_proj])
# pay attention to the wrt parameters to get the extract order for the gradients
updates = [
    (W_conv, W_conv - learning_rate * grad[0]),
    (b_conv, b_conv - learning_rate * grad[1]),
    (W_proj, W_proj - learning_rate * grad[2]),
    (b_proj, b_proj - learning_rate * grad[3]),
]

f_train = theano.function([X, y_true], cost, updates=updates,
                          allow_input_downcast=True)
f_pred = theano.function([X], y_pred,
                         allow_input_downcast=True)


training_history, valid_history = train_networks(f_train, f_pred,
                                        nb_epoch=3, batch_size=128,
                                        X_train=X_train, y_train=y_train,
                                        X_valid=X_valid, y_valid=y_valid)


report_performance(f_pred, X_test, y_test)

plt.figure()
plt.plot(training_history, c='b', label="Training cost")
plt.plot(valid_history, c='r', label="Validation accuracy")
plt.legend()
plt.show()

W_ = W_conv.get_value()
plot_weights4D(W_, colormap = "Blues")
plt.show()

f = theano.function([X], conv_output, allow_input_downcast=True)
X_new = f(X_train[:16])
plot_weights4D(X_new, 'Reds')
plt.show()
