# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:52:21 2019

@author: finlab620
"""

# import the necessary packages
from keras.datasets import fashion_mnist



print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()