from __future__ import print_function, absolute_import, division
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.core.pylabtools import figsize
figsize(12, 4)

import os
os.environ['THEANO_FLAGS'] = 'device=cpu,optimizer=fast_run'
import sys
sys.path.append(os.path.join('/res', 'src'))

import scipy.io as sio
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, merge, Convolution1D, MaxPooling1D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import (plot_genes, report_performance, plot_confusion_matrix,
                   plot_weights, plot_weights4D)
np.random.seed(12082518)

# Load data
d = sio.loadmat('/res/data/seq_data.mat')
X = d['data']
y = d['labels']
# shuffle data
idx = np.random.permutation(X.shape[0])
X = X[idx]
y = y[idx]
# divide into training and testing
n = X.shape[0]
X_train = X[:int(0.8 * n)]
y_train = y[:int(0.8 * n)]

X_test = X[int(0.8 * n):]
y_test = y[int(0.8 * n):]

# You must make sure both training and testing data contain all 4 labels
print("Training data:", X_train.shape, y_train.shape, set(np.argmax(y_train, -1)))
print("Testing data:", X_test.shape, y_test.shape, set(np.argmax(y_test, -1)))

# start plotting
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Training set statistics")
plt.hist(np.argmax(y_train, -1), bins=4)
plt.xticks([0.4, 1.1, 1.8, 2.6], range(4))

plt.subplot(1, 2, 2)
plt.title("Testing set statistics")
plt.hist(np.argmax(y_test, -1), bins=4)
plt.xticks([0.4, 1.1, 1.8, 2.6], range(4))

plt.show()

print('Rad21 peak regions from K562 cell line ')
tmp = X[np.argmax(y, -1) == 0]
plt.figure()
plot_genes(tmp[:10])
plt.show()

print('E2F4 peak regions from K562 cell line')
tmp = X[np.argmax(y, -1) == 1]
plt.figure()
plot_genes(tmp[:10])
plt.show()

print('Nrf1 peak regions from GM12878 cell line')
tmp = X[np.argmax(y, -1) == 2]
plt.figure()
plot_genes(tmp[:10])
plt.show()

print('BRCA1 peak regions from GM12787 cell line')
tmp = X[np.argmax(y, -1) == 3]
plt.figure()
plot_genes(tmp[:10])
plt.show()

# For big model LEARNING RATE should be smaller
LEARNING_RATE = 0.0001
# we have 1024 GB of RAM so why not (pay more attention to
# batch size when you run it on GPU)
BATCH_SIZE = 64

######## [YOUR CODE HERE: modify the network to proceduce best results] ########
input = Input(shape=(1000, 4))
conv = Convolution1D(nb_filter=32, filter_length=3, activation='relu')(input)
pool = MaxPooling1D(pool_length=13, stride=13)(conv)
drop1 = Dropout(0.2)(pool)
lstm = LSTM(output_dim=64, return_sequences=True, consume_less='gpu')(drop1)
drop2 = Dropout(0.5)(lstm)
flat = Flatten()(drop2)
dense = Dense(output_dim=1024, activation='relu')(flat)
output = Dense(output_dim=4, activation='sigmoid')(dense)

model = Model(input=input, output=output)
# choosing optimization algorithm
optim = RMSprop(lr=LEARNING_RATE, rho=0.9, epsilon=1e-8)
# Compile the model to high-performance C/C++ code
model.compile(optimizer=optim, loss='binary_crossentropy',
              metrics=['accuracy'])

name = 'model%d.png' % np.random.randint(10e8)
path = '/tmp/%s' % name
# This command only save the image of your model
plot(model, to_file=path, show_shapes=True, show_layer_names=True)
# Load and review the image (increase both dimension of figsize
# if you see too small images)
plt.figure(figsize=(8, 120))
plt.imshow(mpimg.imread(path))
plt.axis('off')
plt.show()

# Callback
checkpointer = ModelCheckpoint(filepath='seq_example_best_weights.hdf5', verbose=1, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,
                    nb_epoch=3, verbose=1,
                    shuffle=True, validation_split=0.2,
                    callbacks=[checkpointer])

y = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
# the prediction is just probability values, we select the highest probability
# values
y = np.argmax(y, axis=-1)
# convert one-hot encoded y_test to label also
y_test = np.argmax(y_test, axis=-1)
print('Test accuracy:', accuracy_score(y_test, y))
print('Classification report:', classification_report(y_test, y))
plt.figure()
plot_confusion_matrix(confusion_matrix(y_test, y),
                      labels=range(0, 4))
plt.show()
