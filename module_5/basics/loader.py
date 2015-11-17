#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import os
import pickle

__author__ = 'krekle'

# Path to MNIST files
__mnist_path__ = os.path.dirname(os.path.abspath(__file__)) + '\\'


def indexify(x, n):
    if type(x) == list:
        x = numpy.array(x)
    x = x.flatten()
    o_h = numpy.zeros((len(x), n))
    o_h[numpy.arange(len(x)), x] = 1
    return o_h


def mnist(ntrain=60000, ntest=10000, indexed=True):
    data_dir = __mnist_path__
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX / 255.
    teX = teX / 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if indexed:
        trY = indexify(trY, 10)
        teY = indexify(teY, 10)
    else:
        trY = numpy.asarray(trY)
        teY = numpy.asarray(teY)

    return trX, teX, trY, teY


# This loads any collection of flat MNIST cases from a file.
def load_cases(filename='demo_prep', dir=__mnist_path__, nested=True):
    fcases = load_flat_cases(filename, dir)
    return reconstruct_flat_cases(fcases, nested=nested)


def load_flat_cases(filename, dir=__mnist_path__):
    f = open(dir + filename, 'rb')
    return pickle.load(f)


def reconstruct_flat_cases(cases, dims=(28, 28), nested=True):
    labels = numpy.array([[label] for label in cases[1]]) if nested else cases[1]
    images = [reconstruct_image(i, dims=dims) for i in cases[0]] if nested else cases[0]
    return images, labels


def reconstruct_image(flat_list, dims=(28, 28)):
    image = numpy.array(flat_list)
    image = numpy.reshape(image, dims)
    return image
