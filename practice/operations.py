import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

# TensorFlow will define a graph to store variables then this graph is run through session

with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b: %i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants %i" % sess.run(a*b))

# tf.placeholder make a tensor that can take specific values, here it is 16bit integer
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b) # Supports Broadcasting

# feed_dict is used to feed values in variable tensors
with tf.Session() as sess:
    print("Addition with variables: %i" %sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

#1x2 matrix
matrix1 = tf.constant([[3., 3.]])
#2x1 matrix
matrix2 = tf.constant([[2.],[2.]])
#tf.matmul is used to multiply matix, also supports broadcasting
product = tf.matmul(matrix1, matrix2)

#IMPORTANT: None of the above operations are executed untill session is run as it only create the graph

with tf.Session() as sess:
    result = sess.run(product)
    print(result)
