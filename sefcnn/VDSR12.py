import tensorflow as tf
import numpy as np

def model(input_tensor):
#    with tf.device("/gpu:0"):
        weights = []
        tensor = None
        convId = 0
        conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
        conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
        convId += 1
        weights.append(conv_00_w)
        weights.append(conv_00_b)
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))

        for i in range(10):
            conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
            convId += 1
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))

        conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
        conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
        convId += 1
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

        tensor = tf.add(tensor, input_tensor)
        return tensor, weights
