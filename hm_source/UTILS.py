import numpy as np
import tensorflow as tf
import math, os, random
from PIL import Image
from VDSRUNK import BATCH_SIZE
from VDSRUNK import PATCH_SIZE

def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)

def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)

def truncate(input, min, max):
    input = np.where(input > min, input, min)
    input = np.where(input < max, input, max)
    return input

def remap(input):
    input = 16+219/255*input
    #return tf.clip_by_value(input, 16, 235).eval()
    return truncate(input, 16.0, 235.0)

def deremap(input):
    input = (input-16)*255/219
    #return tf.clip_by_value(input, 0, 255).eval()
    return truncate(input, 0.0, 255.0)

def load_file_list(directory):
    list = []
    for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory,y))]:
        list.append(os.path.join(directory,filename))
    return sorted(list)

def get_train_list(lowList, highList):
    assert len(lowList) == len(highList), "low:%d, high:%d"%(len(lowList), len(highList))
    train_list = []
    for i in range(len(lowList)):
        train_list.append([lowList[i], highList[i]])
    return train_list

def prepare_nn_data(train_list, idx_img=None):
    i = np.random.randint(len(train_list)) if (idx_img is None) else idx_img

    input_image  = c_getYdata(train_list[i][0])
    gt_image = c_getYdata(train_list[i][1])

    input_list = []
    gt_list = []
    inputcbcr_list = []

    for idx in range(BATCH_SIZE):

        #crop images to the disired size.
        input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")

        #normalize
        input_imgY = normalize(input_imgY)
        gt_imgY = normalize(gt_imgY)

        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        #inputcbcr_list.append(input_imgCbCr)

    input_list = np.resize(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    #inputcbcr_list = np.resize(inputcbcr_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 2))

    return input_list, gt_list, inputcbcr_list

def getWH(yuvfileName):
    #w_included , h_included = os.path.basename(yuvfileName).split('x')
    #w = w_included.split('_')[-1]
    #h = h_included.split('_')[0]
    wxh = os.path.basename(yuvfileName).split('_')[-2]
    w, h = wxh.split('x')
    return int(w), int(h)

def getYdata(path, size):
    w= size[0]
    h=size[1]
    #print(w,h)
    Yt = np.zeros([h, w], dtype="uint8", order='C')
    with open(path, 'rb') as fp:
        fp.seek(0, 0)
        Yt = fp.read()
        tem = Image.frombytes('L', [w, h], Yt)

        Yt = np.asarray(tem, dtype='float32')

        # for n in range(h):
        #     for m in range(w):
        #         Yt[n, m] = ord(fp.read(1))

    return Yt

def c_getYdata(path):
    return getYdata(path, getWH(path))

def img2y(input_img):
    if np.asarray(input_img).shape[2] == 3:
        input_imgY = input_img.convert('YCbCr').split()[0]
        input_imgCb, input_imgCr = input_img.convert('YCbCr').split()[1:3]

        input_imgY = np.asarray(input_imgY, dtype='float32')
        input_imgCb = np.asarray(input_imgCb, dtype='float32')
        input_imgCr = np.asarray(input_imgCr, dtype='float32')


        #Concatenate Cb, Cr components for easy, they are used in pair anyway.
        input_imgCb = np.expand_dims(input_imgCb,2)
        input_imgCr = np.expand_dims(input_imgCr,2)
        input_imgCbCr = np.concatenate((input_imgCb, input_imgCr), axis=2)

    elif np.asarray(input_img).shape[2] == 1:
        print("This image has one channal only.")
        #If the num of channal is 1, remain.
        input_imgY = input_img
        input_imgCbCr = None
    else:
        print("The num of channal is neither 3 nor 1.")
        exit()
    return input_imgY, input_imgCbCr

def crop(input_image, gt_image, patch_width, patch_height, img_type):
    assert type(input_image) == type(gt_image), "types are different."
    #return a ndarray object
    if img_type == "ndarray":
        in_row_ind   = random.randint(0,input_image.shape[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.shape[1]-patch_height)

        input_cropped = input_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind+patch_width, in_col_ind:in_col_ind+patch_height]

    #return an "Image" object
    elif img_type == "Image":
        in_row_ind   = random.randint(0,input_image.size[0]-patch_width)
        in_col_ind   = random.randint(0,input_image.size[1]-patch_height)

        input_cropped = input_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))
        gt_cropped = gt_image.crop(box=(in_row_ind, in_col_ind, in_row_ind+patch_width, in_col_ind+patch_height))

    return input_cropped, gt_cropped

def save_images(inputY, inputCbCr, size, image_path):
    """Save mutiple images into one single image.

    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    image_path : string.

    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> tl.visualize.save_images(images, [8, 8], 'temp.png')
    """
    def merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img

    inputY = inputY.astype('uint8')
    inputCbCr = inputCbCr.astype('uint8')
    output_concat = np.concatenate((inputY, inputCbCr), axis=3)

    assert len(output_concat) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(output_concat))

    new_output = merge(output_concat, size)

    new_output = new_output.astype('uint8')

    img = Image.fromarray(new_output, mode='YCbCr')
    img = img.convert('RGB')
    img.save(image_path)

def get_image_batch(train_list,offset,batch_size):
    target_list = train_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    inputcbcr_list = []
    for pair in target_list:
        input_img = Image.open(pair[0])
        gt_img = Image.open(pair[1])

        #crop images to the disired size.
        input_img, gt_img = crop(input_img, gt_img, PATCH_SIZE[0], PATCH_SIZE[1], "Image")

        #focus on Y channal only
        input_imgY, input_imgCbCr = img2y(input_img)
        gt_imgY, gt_imgCbCr = img2y(gt_img)

        #input_imgY = normalize(input_imgY)
        #gt_imgY = normalize(gt_imgY)

        input_list.append(input_imgY)
        gt_list.append(gt_imgY)
        inputcbcr_list.append(input_imgCbCr)

    input_list = np.resize(input_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.resize(gt_list, (batch_size, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list, inputcbcr_list

def save_test_img(inputY, inputCbCr, path):
    assert len(inputY.shape) == 4, "the tensor Y's shape is %s"%inputY.shape
    assert inputY.shape[0] == 1, "the fitst component must be 1, has not been completed otherwise.{}".format(inputY.shape)

    inputY = np.squeeze(inputY, axis=0)
    inputY = inputY.astype('uint8')

    inputCbCr = inputCbCr.astype('uint8')

    output_concat = np.concatenate((inputY, inputCbCr), axis=2)
    img = Image.fromarray(output_concat, mode='YCbCr')
    img = img.convert('RGB')
    img.save(path)

def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if((type(hr_image)==type(np.array([]))) or (type(hr_image)==type([]))):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff*diff)
        mse = np.maximum(eps, mse)
        return float(10*math.log10(max_value*max_value/mse))
    else:
        assert len(hr_image.shape)==4 and len(sr_image.shape)==4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10*tf.log(max_value*max_value/mse)/math.log(10)
