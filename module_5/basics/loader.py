#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from numpy import array
import os
import pickle

__author__ = 'krekle'

# Path to MNIST files
_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


def indexify(x, n):
    if type(x) == list:
        x = numpy.array(x)
    x = x.flatten()
    o_h = numpy.zeros((len(x), n))
    o_h[numpy.arange(len(x)), x] = 1
    return o_h


def mnist(ntrain=60000, ntest=10000, indexed=True):
    fd = open(os.path.join(_dir, 'train-images.idx3-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(_dir, 'train-labels.idx1-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(_dir, 't10k-images.idx3-ubyte'))
    loaded = numpy.fromfile(file=fd, dtype=numpy.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(_dir, 't10k-labels.idx1-ubyte'))
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
def load_cases(filename='demo_prep', dir=_dir, nested=True):
    fcases = load_flat_cases(filename, dir)
    images, labels = reconstruct_flat_cases(fcases, nested=nested)

    #img_geometry = (array(images).shape[0])
    #images = array(images).reshape((len(images), 28*28)).astype(float)
    #labels = indexify(array(labels).reshape(len(labels)), 10)
    #return images, indexify(labels, 10)

    # Flatten and reshape images
    images = array(images)
    l = int(images.shape[0])
    w = int(images.shape[1])
    w2 = int(images.shape[2])
    images = images.reshape((l, w*w2)).astype(float)
    return images, indexify(labels, 10)

def load_flat_cases(filename, dir=_dir):
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

# You must get this procedure to work PRIOR to the demo session.  It will be swapped out with something very
# similar during the demo session.

def minor_demo(ann,ignore=0):

    def score_it(classification,k=4):
        params = {"results": str(ignore) + " " + str(classification), "raw": "1","k": k}
        resp = requests.post('http://folk.ntnu.no/valerijf/5/', data=params)
        return resp.text

    def test_it(ann,cases,k=4):
        images,_ = cases
        predictions = ann.blind_test(images)  # Students must write THIS method for their ANN
        print(predictions)
        # return score_it(predictions,k=k)

    demo100 = load_flat_text_cases('demo100_text.txt')
    training_cases = load_flat_text_cases('all_flat_mnist_training_cases_text.txt')
    test_cases = load_flat_text_cases('all_flat_mnist_testing_cases_text.txt')
    print('TEST Results:')
    print('Training set: \n ',test_it(ann,training_cases,4))
    print('Testing set:\n ',test_it(ann,test_cases,4))
    print('Demo 100 set: \n ',test_it(ann,demo100,8))


# For reading flat cases from a TEXT file (as provided by Valerij). This is needed for the demo!
def load_flat_text_cases(filename, dir=_dir):
    f = open(dir + filename, "r")
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()
    x_l = list(map(int, lines[0]))
    x_t = [list(map(int, line)) for line in lines[1:]]
    return x_t, x_l

