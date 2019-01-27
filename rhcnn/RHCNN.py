import tensorflow as tf
import numpy as np


def model(input_tensor):
    #    with tf.device("/gpu:0"):
    arrayNum = [64, 128, 256, 128, 64, 1]
    # arrayNum = [64, 32, 1]
    weights = []
    tensor = input_tensor
    convId = 0
    tem = None

    conv_w_min = tf.get_variable('conv_%02d_Middle_w' % (convId), [3, 3, 1, 64],
                                 initializer=tf.contrib.layers.xavier_initializer())
    conv_b_min = tf.get_variable('conv_%02d_Middle_b' % (convId), [64], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, conv_w_min)
    tf.add_to_collection(tf.GraphKeys.BIASES, conv_b_min)
    weights.append(conv_w_min)
    weights.append(conv_b_min)
    tensorMin3 = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w_min, strides=[1, 1, 1, 1], padding='SAME'), conv_b_min))

    conv_wL = tf.get_variable('conv_%02d_Left_w' % (convId), [3, 3, 64, 64],
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_bL = tf.get_variable('conv_%02d_Left_b' % (convId), [64], initializer=tf.constant_initializer(0))
    weights.append(conv_wL)
    weights.append(conv_bL)

    tensorL = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(tensorMin3, conv_wL, strides=[1, 1, 1, 1], padding='SAME'), conv_bL))
    conv_wR = tf.get_variable('conv_%02d_Right_w' % (convId), [3, 3, 64, 64],
                              initializer=tf.contrib.layers.xavier_initializer())
    conv_bR = tf.get_variable('conv_%02d_Right_b' % (convId), [64], initializer=tf.constant_initializer(0))
    weights.append(conv_wR)
    weights.append(conv_bR)
    tensorR = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(tensorMin3, conv_wR, strides=[1, 1, 1, 1], padding='SAME'), conv_bR))
    tensorConcat = tf.concat([tensorL, tensorR], axis=3)  # axis = 0???
    conv_1x1 = tf.get_variable('conv_%02d_1x1_w' % (convId), [1, 1, 128, 64],
                               initializer=tf.contrib.layers.xavier_initializer())
    convb_1x1 = tf.get_variable('conv_%02d_1x1_b' % (convId), [64],
                               initializer=tf.constant_initializer(0))
    weights.append(conv_1x1)
    weights.append(convb_1x1)
    tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensorConcat, conv_1x1, strides=[1, 1, 1, 1], padding='SAME'), convb_1x1))
    convId += 1

    for i in range(1, 6):
        conv_w_min = tf.get_variable('conv_%02d_Middle_w' % (convId), [3, 3, arrayNum[i - 1], arrayNum[i]],
                                     initializer=tf.contrib.layers.xavier_initializer())
        conv_b_min = tf.get_variable('conv_%02d_Middle_b' % (convId), [arrayNum[i]],
                                     initializer=tf.constant_initializer(0))

        weights.append(conv_w_min)
        weights.append(conv_b_min)
        tensorMin = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w_min, strides=[1, 1, 1, 1], padding='SAME'), conv_b_min))

        conv_wL = tf.get_variable('conv_%02d_Left_w' % (convId), [3, 3, arrayNum[i], arrayNum[i]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_bL = tf.get_variable('conv_%02d_Left_b' % (convId), [arrayNum[i]], initializer=tf.constant_initializer(0))
        weights.append(conv_wL)
        weights.append(conv_bL)
        tensorL = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(tensorMin, conv_wL, strides=[1, 1, 1, 1], padding='SAME'), conv_bL))
        conv_wR = tf.get_variable('conv_%02d_Right_w' % (convId), [3, 3, arrayNum[i], arrayNum[i]],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv_bR = tf.get_variable('conv_%02d_Right_b' % (convId), [arrayNum[i]], initializer=tf.constant_initializer(0))
        weights.append(conv_wR)
        weights.append(conv_bR)
        tensorR = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(tensorMin, conv_wR, strides=[1, 1, 1, 1], padding='SAME'), conv_bR))
        tensorConcat = tf.concat([tensorL, tensorR], axis=3)  # axis = 0???
        conv_1x1 = tf.get_variable('conv_%02d_1x1_w' % (convId), [1, 1, 2 * arrayNum[i], arrayNum[i]],
                                   initializer=tf.contrib.layers.xavier_initializer())
        convb_1x1 = tf.get_variable('conv_%02d_1x1_b' % (convId), [arrayNum[i]],
                                   initializer=tf.constant_initializer(0))
        weights.append(conv_1x1)
        weights.append(convb_1x1)
        if i == 3:
            tem = tensor
            temw = conv_1x1

        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensorConcat, conv_1x1, strides=[1, 1, 1, 1], padding='SAME'), convb_1x1))
        convId += 1

    conv_w_end = tf.get_variable("conv_%02d_end_w" % (convId), [1, 1, 1, 1],
                                 initializer=tf.contrib.layers.xavier_initializer())
    conv_b_end = tf.get_variable("conv_%02d_end_b" % (convId), [1], initializer=tf.constant_initializer(0))
    weights.append(conv_w_end)
    weights.append(conv_b_end)
    tensor_end = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w_end, strides=[1, 1, 1, 1], padding='SAME'), conv_b_end)

    # tensor_end = tf.add(input_tensor, tensor_end)
    # return tensor_end, [weights, input_tensor, tensorMin3, tem, temw, tensor_end]
    return tensor_end, weights