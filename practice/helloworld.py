import tensorflow as tf

# It will print Hello with tensorflow

# Creating a constant tensor
hello = tf.constant("Hello, TensorFlow!")

# We need to start tf session to compute using tensorflow as it is highly optimized for distributed computing
sess = tf.Session()

print(sess.run(hello))
