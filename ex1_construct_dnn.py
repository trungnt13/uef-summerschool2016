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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import h5py # for loading data

from utils import (plot_images, Progbar, plot_confusion_matrix, plot_weights,
                   glorot_uniform, validate_deep_network)

dataset = h5py.File(os.path.join(DATA_DIR, 'mnist.h5'), 'r')
for key, value in dataset.iteritems():
    print('Name:%s, Shape:%s, Dtype:%s' % (key, value.shape, value.dtype))
# Load the training data
X_train = dataset['X_train'].value
y_train = dataset['y_train'].value
# Load validation data
X_valid = dataset['X_valid'].value
y_valid = dataset['y_valid'].value
# Load test data
X_test = dataset['X_test'].value
y_test = dataset['y_test'].value

# pick randomly 16 images from training data
random_choices = np.random.choice(np.arange(X_train.shape[0]),
                                  size=16, replace=False)
X_sampled = X_train[random_choices]
y_samples = y_train[random_choices]

# start plotting
plt.figure()
_ = plot_images(X_sampled)
plt.show()
print(y_samples)

# our features are stored in a tensor (nb_samples, nb_row, nb_col)
X = T.tensor3(name='X', dtype='float32')
y_true = T.ivector() # our output is integer vector (i.e. the number 1, 2, 3, 4, 5, 6 ...)

# Our parameters
nb_features = np.prod(X_train.shape[1:]) # 784
nb_classes = 10 # 10 different digits

# Continue Initlaize parameters for your deep network here
# (it must be more than 2 layers)
W = [
    # The first layer is 512 units
    theano.shared(glorot_uniform((nb_features, 256)), name='W_in_1'),
    ######## [Your code here: Happy Coding] ########
    # The last layer (before output layer) is 64 units
    theano.shared(glorot_uniform((64, nb_classes)), name='W_out')
]
# initialize your bias for each layers here
B = [
    ######## [Your code here: Happy Coding] ########
]
# Choose activation for each layers
activate_functions = [
    ######## [Your code here: Happy Coding] ########
]

validate_deep_network(W, B, activate_functions)

# activation just a linear combination of features and parameters (weights)
activation = T.flatten(X, outdim=2) # first value (inputs)
for w, b, f in zip(W, B, activate_functions):
    ######## [Your code here: Happy Coding] ########
    pass
# our prediction is final activation
y_pred = activation
# We use categorical_crossentropy as objective function
cost = T.mean(T.nnet.categorical_crossentropy(y_pred, y_true))
# gradient descent
gradient = T.grad(cost=cost, wrt=W)
# Create your updates (Gradient descent) here for EACH weight and bias,
# it is a mapping (weight, new_weight)
learning_rate = theano.shared(np.cast['float32'](0.1), name='learning_rate')
updates = [
    ######## [Your code here: Happy Coding] ########
]
# create function for training and making prediction
f_train = theano.function(inputs=[X, y_true], outputs=cost,
                        updates=updates,
                        allow_input_downcast=True)
f_predict = theano.function(inputs=[X], outputs=y_pred,
                            allow_input_downcast=True)

NB_EPOCH = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.1

learning_rate.set_value(np.cast['float32'](LEARNING_RATE))
training_history = []
valid_history = []
for epoch in range(NB_EPOCH):
    prog = Progbar(target=X_train.shape[0])
    n = 0
    history = []
    while n < X_train.shape[0]:
        start = n
        end = min(n + BATCH_SIZE, X_train.shape[0])
        c = f_train(X_train[start:end], y_train[start:end])
        prog.title = 'Epoch: %.2d, Cost: %.4f' % (epoch + 1, c)
        prog.add(end - start)
        n += BATCH_SIZE
        history.append(c)
    # end of epoch, start validating
    y = np.argmax(f_predict(X_valid), axis=-1)
    accuracy = accuracy_score(y_valid, y)
    print('Validation accuracy:', accuracy)
    # save history
    training_history.append(np.mean(history))
    valid_history.append(accuracy)


y = np.argmax(f_predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, y)
print('Test accuracy:', accuracy)
print('Classification report:')
print(classification_report(y_test, y))

plt.figure()
plot_confusion_matrix(confusion_matrix(y_test, y),
                      labels=range(1, 11))
plt.show()

plt.figure()
plt.plot(training_history, c='b', label="Training cost")
plt.plot(valid_history, c='r', label="Validation accuracy")
plt.legend()
plt.show()

plt.figure()
for i, w in enumerate(W):
    plt.subplot(len(W), 1, i + 1)
    plot_weights(w.get_value(), keep_aspect=False)
plt.show()
