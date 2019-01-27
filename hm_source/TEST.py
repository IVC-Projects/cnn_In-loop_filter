import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import numpy as np
import tensorflow as tf
import os, time
from SEFCNN import model   #网络

from UTILS import *

I_MODEL_PATH = r"..........."  #模型路径
P_MODEL_PATH = r"..........."
B_MODEL_PATH = r"..........."  

def prepare_test_data(fileOrDir):
    original_ycbcr = []
    gt_y = []
    fileName_list = []
    imgCbCr = 0
    fileName_list.append(fileOrDir)
    imgY = np.reshape(fileOrDir,(1, len(fileOrDir), len(fileOrDir[0]), 1))
    imgY = normalize(imgY)
    original_ycbcr.append([imgY, imgCbCr])
    return original_ycbcr, gt_y, fileName_list

def test_all_ckpt(modelPath, fileOrDir,flags):
    tf.reset_default_graph()

    tf.logging.warning(modelPath)
    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights = shared_model(input_tensor)
        output_tensor = tf.clip_by_value(output_tensor, 0., 1.)
        output_tensor = output_tensor * 255

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)

        for ckpt in ckptFiles:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            if flags==0:
                if epoch != 555:
                    continue
            elif flags==1:
                if epoch!= 555:
                    continue
            else:
                if epoch != 555:
                    continue

            tf.logging.warning("epoch:%d\t"%epoch)
            saver.restore(sess,os.path.join(modelPath,ckpt))
            total_imgs = len(fileName_list)
            for i in range(total_imgs):
                imgY = original_ycbcr[i][0]
                out = sess.run(output_tensor, feed_dict={input_tensor: imgY})
                out = np.reshape(out, (out.shape[1], out.shape[2]))
                out = np.around(out)
                out = out.astype('int')
                out = out.tolist()
                return out

def modelI(inp):
    tf.logging.warning("python, in I")
    i = test_all_ckpt(I_MODEL_PATH, inp,0)
    return i
def modelP(inp):
    tf.logging.warning("python, in p")
    p = test_all_ckpt(P_MODEL_PATH, inp,1)
    return p
def modelB(inp):
    tf.logging.warning("python, in B")
    b = test_all_ckpt(B_MODEL_PATH, inp,2)
    return b
