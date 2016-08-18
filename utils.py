from __future__ import print_function, division, absolute_import

import sys
import math
import time

import numpy as np
import theano
from matplotlib import pyplot as plt
try:
    import seaborn
except:
    pass


# ===========================================================================
# Progress bar
# ===========================================================================
class Progbar(object):

    '''
    This function is adpated from: https://github.com/fchollet/keras
    Original work Copyright (c) 2014-2015 keras contributors
    Modified work Copyright 2016-2017 TrungNT
    '''

    def __init__(self, target, title=''):
        '''
            @param target: total number of steps expected
        '''
        self.width = 39
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.title = title

    def update(self, current, values=[]):
        '''
            @param current: index of current step
            @param values: list of tuples (name, value_for_last_step).
            The progress bar will display averages for these values.
        '''
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()

        prev_total_width = self.total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%s %%%dd/%%%dd [' % (self.title, numdigits, numdigits)
        bar = barstr % (current, self.target)
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if current < self.target:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        self.total_width = len(bar)

        if current:
            time_per_unit = (now - self.start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.target - current)
        info = ''
        if current < self.target:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - self.start)
        for k in self.unique_values:
            info += ' - %s:' % k
            if type(self.sum_values[k]) is list:
                avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
                if abs(avg) > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            else:
                info += ' %s' % self.sum_values[k]

        self.total_width += len(info)
        if prev_total_width > self.total_width:
            info += ((prev_total_width - self.total_width) * " ")

        sys.stdout.write(info)
        if current >= self.target:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)


# ===========================================================================
# Plot genes
# ===========================================================================
def plot_genes(matrices):
    colormap = 'Reds'
    if matrices.ndim == 3:
        matrices = [i for i in matrices]
    elif not isinstance(matrices, (tuple, list)):
        matrices = [matrices]
    nrow = int(math.ceil(len(matrices) / 10))
    # ====== test ====== #
    for i, matrix in enumerate(matrices):
        ax = plt.subplot(nrow, 10, i + 1)
        if matrix.ndim != 2:
            raise ValueError("Only accept matrix with 2-dimensions, "
                            "but the given input has %d-dimensions" % matrix.ndim)
        # ax.set_aspect('equal', 'box')
        img = ax.pcolorfast(matrix, cmap=colormap, alpha=0.9)
        # plt.colorbar(img, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    plt.show(block=True)


# ===========================================================================
# Plot images
# ===========================================================================
def resize_images(x, shape):
    from scipy.misc import imresize

    reszie_func = lambda x, shape: imresize(x, shape, interp='bilinear')
    if x.ndim == 4:
        def reszie_func(x, shape):
            # x: 3D
            # The color channel is the first dimension
            tmp = []
            for i in x:
                tmp.append(imresize(i, shape).reshape((-1,) + shape))
            return np.swapaxes(np.vstack(tmp).T, 0, 1)

    imgs = []
    for i in x:
        imgs.append(reszie_func(i, shape))
    return imgs


def tile_raster_images(X, tile_shape=None, tile_spacing=(2, 2), spacing_value=0.):
    ''' This function create tile of images

    Parameters
    ----------
    X : 3D-gray or 4D-color images
        for color images, the color channel must be the second dimension
    tile_shape : tuple
        resized shape of images
    tile_spacing : tuple
        space betwen rows and columns of images
    spacing_value : int, float
        value used for spacing

    '''
    if X.ndim == 3:
        img_shape = X.shape[1:]
    elif X.ndim == 4:
        img_shape = X.shape[2:]
    else:
        raise ValueError('Unsupport %d dimension images' % X.ndim)
    if tile_shape is None:
        tile_shape = img_shape
    if tile_spacing is None:
        tile_spacing = (2, 2)

    if img_shape != tile_shape:
        X = resize_images(X, tile_shape)
    else:
        X = [np.swapaxes(x.T, 0, 1) for x in X]

    n = len(X)
    n = int(np.ceil(np.sqrt(n)))

    # create spacing
    rows_spacing = np.zeros_like(X[0])[:tile_spacing[0], :] + spacing_value
    nothing = np.vstack((np.zeros_like(X[0]), rows_spacing))
    cols_spacing = np.zeros_like(nothing)[:, :tile_spacing[1]] + spacing_value

    # ====== Append columns ====== #
    rows = []
    for i in range(n): # each rows
        r = []
        for j in range(n): # all columns
            idx = i * n + j
            if idx < len(X):
                r.append(np.vstack((X[i * n + j], rows_spacing)))
            else:
                r.append(nothing)
            if j != n - 1:   # cols spacing
                r.append(cols_spacing)
        rows.append(np.hstack(r))
    # ====== Append rows ====== #
    img = np.vstack(rows)[:-tile_spacing[0]]
    return img


def plot_images(X, tile_shape=None, tile_spacing=None, fig=None, title=None):
    '''
    x : 2D-gray or 3D-color images, or list of (2D, 3D images)
        for color image the color channel is second dimension
    '''
    from matplotlib import pyplot as plt
    if not isinstance(X, (tuple, list)):
        X = [X]
    if not isinstance(title, (tuple, list)):
        title = [title]

    n = int(np.ceil(np.sqrt(len(X))))
    for i, (x, t) in enumerate(zip(X, title)):
        if x.ndim == 3 or x.ndim == 2:
            cmap = plt.cm.Greys_r
        elif x.ndim == 4:
            cmap = None
        else:
            raise ValueError('NO support for %d dimensions image!' % x.ndim)

        x = tile_raster_images(x, tile_shape, tile_spacing)
        if fig is None:
            fig = plt.figure()
        subplot = fig.add_subplot(n, n, i + 1)
        subplot.imshow(x, cmap=cmap)
        if t is not None:
            subplot.set_title(str(t), fontsize=12)
        subplot.axis('off')

    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels):
    from matplotlib import pyplot as plt

    title = 'Confusion matrix'
    cmap = plt.cm.Blues

    # column normalize
    if np.max(cm) > 1:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm
    axis = plt.gca()

    im = axis.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    axis.set_title(title)
    # axis.get_figure().colorbar(im)

    tick_marks = np.arange(len(labels))
    axis.set_xticks(tick_marks)
    axis.set_yticks(tick_marks)
    axis.set_xticklabels(labels, rotation=90, fontsize=13)
    axis.set_yticklabels(labels, fontsize=13)
    axis.set_ylabel('True label')
    axis.set_xlabel('Predicted label')
    # Turns off grid on the left Axis.
    axis.grid(False)

    plt.colorbar(im, ax=axis)

    # axis.tight_layout()
    return axis


def plot_weights(x, keep_aspect=True):
    '''
    Parameters
    ----------
    x : np.ndarray
        2D array
    ax : matplotlib.Axis
        create by fig.add_subplot, or plt.subplots
    colormap : str
        colormap alias from plt.cm.Greys = 'Greys' ('spectral')
        plt.cm.gist_heat
    colorbar : bool, 'all'
        whether adding colorbar to plot, if colorbar='all', call this
        methods after you add all subplots will create big colorbar
        for all your plots
    path : str
        if path is specified, save png image to given path

    Notes
    -----
    Make sure nrow and ncol in add_subplot is int or this error will show up
     - ValueError: The truth value of an array with more than one element is
        ambiguous. Use a.any() or a.all()

    Example
    -------
    >>> x = np.random.rand(2000, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(2, 2, 1)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 2)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 3)
    >>> dnntoolkit.visual.plot_weights(x, ax)
    >>> ax = fig.add_subplot(2, 2, 4)
    >>> dnntoolkit.visual.plot_weights(x, ax, path='/Users/trungnt13/tmp/shit.png')
    >>> plt.show()
    '''
    from matplotlib import pyplot as plt

    if x.ndim > 2:
        raise ValueError('No support for > 2D')
    elif x.ndim == 1:
        x = x[:, None]

    ax = plt.gca()
    if keep_aspect:
        ax.set_aspect('equal', 'box')
    # ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(str(x.shape), fontsize=6)
    img = ax.pcolorfast(x, cmap='Greys', alpha=0.8)
    plt.grid(True)

    plt.colorbar(img, ax=ax)

    return ax


def plot_weights4D(x):
    '''
    Example
    -------
    >>> # 3D shape
    >>> x = np.random.rand(32, 28, 28)
    >>> dnntoolkit.visual.plot_conv_weights(x)
    '''

    shape = x.shape
    if len(shape) != 4:
        raise ValueError('This function only support 4D weights matrices')

    fig = plt.figure()
    imgs = []
    for i in range(shape[0]):
        imgs.append(tile_raster_images(x[i], tile_spacing=(3, 3)))

    ncols = int(np.ceil(np.sqrt(shape[0])))
    nrows = int(ncols)

    count = 0
    for i in range(nrows):
        for j in range(ncols):
            count += 1
            # skip
            if count > shape[0]:
                continue

            ax = fig.add_subplot(nrows, ncols, count)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            # image data: no idea why pcolorfast flip image vertically
            img = ax.pcolorfast(imgs[count - 1][::-1, :], cmap='Reds', alpha=0.9)

    plt.tight_layout()
    # colorbar
    axes = fig.get_axes()
    fig.colorbar(img, ax=axes)

    return fig
