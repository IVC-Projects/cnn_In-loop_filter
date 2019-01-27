import tensorflow as tf
import numpy as np
#PATCH_SIZE = (35, 35)

def seblock(temp_tensor, convId, weights):
	t_tensor1 = None
	t_tensor2 = None
	t_tensor3 = None
	conv_secondID = 0

	
	for i in range(3):
		if(i==0):
			conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
			conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			#t_tensor1 = batch_norm_layer(t_tensor1, TRAIN_PHASE, "conv_%02d_%02d" % (convId,conv_secondID))    #bn,scale
			t_tensor1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
		elif(i==1):
			conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
			conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			#t_tensor1 = batch_norm_layer(t_tensor1, TRAIN_PHASE, "conv_%02d_%02d" % (convId,conv_secondID))    #bn,scale
			t_tensor1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(t_tensor1, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
		else:
			# no relu
		  conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
		  conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
		  weights.append(conv_w)
		  weights.append(conv_b)
		  #t_tensor1 = batch_norm_layer(t_tensor1, TRAIN_PHASE, "conv_%02d_%02d" % (convId,conv_secondID))    #bn,scale
		  t_tensor1 = tf.nn.bias_add(tf.nn.conv2d(t_tensor1, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)
		conv_secondID += 1
		
	#Now t_tensor1 has passed through 3x3,64 three times
	
	#fetch t_tensor2  by globel average pooling 
	#t_tensor2 = tf.nn.avg_pool(t_tensor1, ksize=[1, PATCH_SIZE[0], PATCH_SIZE[1], 1], strides=[1,1,1,1], padding='SAME')
	t_tensor2 = tf.reduce_mean(t_tensor1,[1, 2])
	t_tensor2 = tf.reshape(t_tensor2, [t_tensor1.shape[0], 1, 1, t_tensor1.shape[-1]])

	
	#1x1_down
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [1,1,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/64)))
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
	conv_secondID += 1
	weights.append(conv_w)
	weights.append(conv_b)
	#relu
	t_tensor2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(t_tensor2, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
	
	#1x1_up
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [1,1,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/64)))
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
	conv_secondID += 1
	weights.append(conv_w)
	weights.append(conv_b)
	#sigmoid
	t_tensor2 = tf.sigmoid(tf.nn.bias_add(tf.nn.conv2d(t_tensor2, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
	
	#gain t_tensor3 after 1x1,64 ,no relu
	conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId,conv_secondID), [1,1,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/64)))
	conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId,conv_secondID), [64], initializer=tf.constant_initializer(0))
	conv_secondID += 1
	weights.append(conv_w)
	weights.append(conv_b)
	t_tensor3 = tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)
	#t_tensor3 = batch_norm_layer(t_tensor3, TRAIN_PHASE, "conv_%02d_%02d" % (convId,conv_secondID))    #bn,scale
	
	#t_tensor2 * t_tensor1 + t_tensor3
	t_tensor1 = tf.nn.relu( tf.add(tf.multiply(t_tensor1, t_tensor2), t_tensor3))
	
	return t_tensor1

def model(input_tensor):
#VDSR15_SE_VDSR
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

        for i in range(14):
            conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
            convId += 1
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
        
        tensor = seblock(tensor, convId, weights)
        convId += 1
        '''for i in range(7):
            conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
            conv_b = tf.get_variable("conv_%02d_b" % (convId), [64], initializer=tf.constant_initializer(0))
            convId += 1
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))'''
            
        conv_w = tf.get_variable("conv_%02d_w" % (convId), [3,3,64,1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
        conv_b = tf.get_variable("conv_%02d_b" % (convId), [1], initializer=tf.constant_initializer(0))
        convId += 1
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b)

        tensor = tf.add(tensor, input_tensor)


        return tensor, weights
