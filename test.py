from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import time
import codecs
import sys
import functools
import math
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.param_attr import ParamAttr
from PIL import Image, ImageEnhance
import argparse
import matplotlib.pyplot as plt

target_size = [3, 224, 224]
mean_rgb = [127.5, 127.5, 127.5]
data_dir = "../datasets/img1.0/"
eval_file = "eval.txt"
use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "./freeze-model-zhedang-1.1"
paddle.enable_static()
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)
# print(fetch_targets)


def crop_image(img, target_size):
    width, height = img.size
    w_start = (width - target_size[2]) / 2
    h_start = (height - target_size[1]) / 2
    w_end = w_start + target_size[2]
    h_end = h_start + target_size[1]
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def resize_img(img, target_size):
    ret = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return ret


def read_image(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = crop_image(img, target_size)
    img = np.array(img).astype('float32')
    img -= mean_rgb
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    img = img[np.newaxis,:]
    return img

def show_image(img_path):
    img = Image.open(img_path)
    plt.figure(img_path)
    plt.imshow(img)
    plt.show()

def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    print(label)
    return np.argmax(label)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_path', default='img/test.jpg', type=str, help='file path of test img')
    args = parser.parse_args()
    result = infer(args.img_path)
    print('predict result is ' + str(result))
    show_image(args.img_path)


# 0	dropframes
# 1	zhechang
# 2	zhedang
