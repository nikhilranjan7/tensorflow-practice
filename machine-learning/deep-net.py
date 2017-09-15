import tensorflow as tf
'''
Feed Forward network:
input > weights > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

Backpropagation:
compare output to intended output > cost function (e.g., cross entropy)
opitmization function (optimizer) > minimize cost(SGD, AdaGrad, AdamOptimizer)
manipulate weight accordingly

feed forward + backpropagation = epoch
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
