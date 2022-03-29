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

target_size = [3, 224, 224]
mean_rgb = [127.5, 127.5, 127.5]
data_dir = "../datasets/img2.2/"
eval_file = "eval.txt"
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "./freeze-model-zhedang-2.2"
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


def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    return np.argmax(label)


def eval_all():
    eval_file_path = os.path.join(data_dir, eval_file)
    total_count = 0
    right_count = 0


    chaidi_count = 0
    chaotian_count = 0
    dropframes_count = 0
    zhechang_count = 0
    zhedang_count = 0

    chaidi_right_count = 0
    chaotian_right_count = 0
    dropframes_right_count = 0
    zhechang_right_count = 0
    zhedang_right_count = 0

# 0	chaodi
# 1	chaotian
# 2	dropframes
# 3	zhechang
# 4	zhedang

    with codecs.open(eval_file_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        t1 = time.time()

        
        for line in lines:
            total_count += 1
            parts = line.strip().split()


            result = infer(parts[0])
            # print("infer result:{0} answer:{1}".format(result, parts[1]))
            if parts[1] == '0':
                chaidi_count += 1
            elif parts[1] == '1':
                chaotian_count += 1
            elif parts[1] == '2':
                dropframes_count += 1
            elif parts[1] == '3':
                zhechang_count += 1
            elif parts[1] == '4':
                zhedang_count += 1
            
            if str(result) == parts[1]:
                right_count += 1
                if parts[1] == '0':
                    chaidi_right_count += 1
                elif parts[1] == '1':
                    chaotian_right_count += 1
                elif parts[1] == '2':
                    dropframes_right_count += 1
                elif parts[1] == '3':
                    zhechang_right_count += 1
                elif parts[1] == '4':
                    zhedang_right_count += 1
        period = time.time() - t1
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period, right_count / total_count))

        print("class 0 chaodi: eval count:{0}/{1} predict accuracy:{2}".format(chaidi_right_count, chaidi_count, chaidi_right_count / chaidi_count))
        print("class 1 chaotian: eval count:{0}/{1} predict accuracy:{2}".format(chaotian_right_count, chaotian_count, chaotian_right_count / chaotian_count))
        print("class 2 dropframes: eval count:{0}/{1} predict accuracy:{2}".format(dropframes_right_count, dropframes_count, dropframes_right_count / dropframes_count))
        print("class 3 zhechang: eval count:{0}/{1} predict accuracy:{2}".format(zhechang_right_count, zhechang_count, zhechang_right_count / zhechang_count))
        print("class 4 zhedang: eval count:{0}/{1} predict accuracy:{2}".format(zhedang_right_count, zhedang_count, zhedang_right_count / zhedang_count))


if __name__ == '__main__':
    eval_all()

# 0	chaodi
# 1	chaotian
# 2	dropframes
# 3	zhechang
# 4	zhedang