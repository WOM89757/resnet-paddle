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
from PIL import Image, ImageEnhance
import argparse
import matplotlib.pyplot as plt

from eval import read_image

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
data_dir = "../datasets/img3.2.1/"
label_file = "label_list.txt"
save_freeze_dir = "./freeze-model-qc-1.2.1"
# save_freeze_dir = "./freeze-model-zhedang-2.3"
# save_freeze_dir = "./freeze-model-zhedang-2.3.1"
# save_freeze_dir = "./freeze-model-zhedang-2.4"
paddle.enable_static()
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)
# print(fetch_targets)

def show_image(img_path):
    img = Image.open(img_path)
    plt.figure(img_path)
    plt.imshow(img)
    plt.show()

def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    np.set_printoptions(suppress=True)
    print(label)
    return np.argmax(label), label[0][0][np.argmax(label)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_path', default='img/test.jpg', type=str, help='file path of test img')
    args = parser.parse_args()
    label_dict = {}
    label_file_path = os.path.join(data_dir, label_file)
    for line in open(label_file_path):
        s = line.splitlines()[0].split('\t')
        label_dict[s[0]] = s[1]

    result_label, result = infer(args.img_path)
    print('predict result is ' + str(result_label) + ' ' + label_dict[str(result_label)] + ': ' + str(result))
    show_image(args.img_path)


# qc 1.1
# 0	diuzhen01
# 1	diuzhen02
# 2	diuzhen03
# 3	zhedang01
# 4	zhedang02
# 5	zhedang03
# 6	zhedang04

# 0	chaodi
# 1	chaotian
# 2	dropframes
# 3	zhechang
# 4	zhedang

#2.3
# 0	exception
# 1	zhechang

#yichang 1.0
# 0	chaodi
# 1	dropframes
# 2	zhedang