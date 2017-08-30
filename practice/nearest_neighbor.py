import numpy as np
import tensorflow as tf

#Importing MNIST data, it is a database of handwritten digits
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training
Xte, Yte = mnist.test.next_batch(200)   #200 for testing

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

#Nearest Neighbor implementation
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

pred = tf.arg_min(distance, 0)
accuracy = 0

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#loop over test data
for i in range(len(Xte)):
    # Get Nearest Neighbor
    nn_index = sess.run(pred, feed_dict={xtr:Xtr, xte: Xte[i:]})
    # get class label and compare it to its true label
    print("Test",i,"Prediction:",np.argmax(Ytr[nn_index]),"True Class:",np.argmax(Yte[i]))
    # Calculate accuracy
    if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
        accuracy += 1./len(Xte)
print("Done!\nAccuracy:",accuracy)
