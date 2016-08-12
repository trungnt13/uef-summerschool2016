
# making inline plot
%matplotlib inline
# make our code python2 and python3 compatible
from __future__ import print_function, absolute_import, division

# use appropriate matplotlib backend for ipython notebook
import matplotlib
matplotlib.use('Agg')
from IPython.core.pylabtools import figsize
figsize(12, 4)
# figures and plot library
from matplotlib import pyplot as plt

# don't need to care about this
import os
import sys
os.environ['THEANO_FLAGS'] = "device=cpu,optimizer=fast_run"
DATA_DIR = os.path.join('/res', 'data')
# path to our libraries source code
sys.path.append(os.path.join('/res', 'src'))

import scipy.io as sio
import numpy as np

# computation libraries
import theano
from theano import tensor as T
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import h5py # for loading data

# some utilities I write for you for easily plotting stuffs
from utils import plot_images, Progbar, plot_confusion_matrix, plot_weights

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

# start plotting
plt.figure()
plt.subplot(1, 3, 1)
plt.title("Training set statistics")
plt.hist(y_train, bins=10)

plt.subplot(1, 3, 2)
plt.title("Validation set statistics")
plt.hist(y_valid, bins=10)

_ = plt.subplot(1, 3, 3)
plt.title("Test set statistics")
plt.hist(y_test, bins=10)
plt.show()



def glorot_uniform(shape, gain=1.0):
    if len(shape) < 2:
        shape = (1,) + tuple(shape)
    n1, n2 = shape[:2]
    receptive_field_size = np.prod(shape[2:])

    std = gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
    a = 0.0 - np.sqrt(3) * std
    b = 0.0 + np.sqrt(3) * std
    return np.cast['float32'](
        np.random.uniform(low=a, high=b, size=shape))

# our features are stored in a tensor (nb_samples, nb_row, nb_col)
X = T.tensor3(name='X', dtype='float32')
y_true = T.ivector() # our output is integer vector (i.e. the number 1, 2, 3, 4, 5, 6 ...)

# Our parameters
nb_features = np.prod(X_train.shape[1:]) # 784
nb_classes = 10 # 10 different digits
W_init = glorot_uniform((nb_features, nb_classes))
W = theano.shared(W_init, name='W')
b = theano.shared(np.zeros(shape=(nb_classes,), dtype='float32'), name='bias')

# activation just a linear combination of features and parameters (weights)
# Don't forget to add the bias
activation = T.dot(T.flatten(X, outdim=2), W) + b
# softmax function "smash" to activation to the probability value (confident value) for each digits
y_pred = T.nnet.softmax(activation)

# We use categorical_crossentropy as objective function
cost = T.mean(T.nnet.categorical_crossentropy(y_pred, y_true))

# gradient descent
W_gradient, b_gradient = T.grad(cost=cost, wrt=[W, b])
# we have to cast the update to float32 to make the type of weights consistent
learning_rate = theano.shared(np.cast['float32'](0.1), name='learning_rate')
update = [(W, W - W_gradient * learning_rate),
          (b, b - b_gradient * learning_rate)]
# create function for training and making prediction
f_train = theano.function(inputs=[X, y_true], outputs=cost,
                          updates=update,
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
plt.subplot(2, 1, 1)
plot_weights(W_init, keep_aspect=False)
plt.subplot(2, 1, 2)
plot_weights(W.get_value(), keep_aspect=False)
plt.show()
