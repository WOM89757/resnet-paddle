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
save_freeze_dir = "./freeze-model-zhedang-2.2"
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
    return np.argmax(label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'img_path', default='img/test.jpg', type=str, help='file path of test img')
    args = parser.parse_args()
    result = infer(args.img_path)
    print('predict result is ' + str(result))
    show_image(args.img_path)

# 0	chaodi
# 1	chaotian
# 2	dropframes
# 3	zhechang
# 4	zhedang
