from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label

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
data_dir = "../datasets/img3.2/"
eval_file = "eval.txt"
label_file = "label_list.txt"
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "./freeze-model-qc-1.2"
# save_freeze_dir = "./freeze-model-yichang-1.1"
# save_freeze_dir = "./freeze-model-zhedang-2.3"
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
    label_file_path = os.path.join(data_dir, label_file)

    label_dict = {}
    label_count = {}
    label_right_count = {}
    right_count = 0
    total_count = 0
    for line in open(label_file_path):
        s = line.splitlines()[0].split('\t')
        label_dict[s[0]] = s[1]
        label_right_count[s[0]] = 0
    # print(label_dict)

    with codecs.open(eval_file_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        t1 = time.time()

        
        for line in lines:
            total_count += 1
            parts = line.strip().split()


            result = infer(parts[0])
            # print("infer result:{0} answer:{1}".format(result, parts[1]))
            if parts[1] in label_count:
                label_count[parts[1]] = label_count[parts[1]] + 1
            else :
                label_count[parts[1]] = 1
            # print(label_count)
            if str(result) == parts[1]:
                right_count += 1
                if parts[1] in label_right_count:
                    label_right_count[parts[1]] = label_right_count[parts[1]] + 1
                else :
                    label_right_count[parts[1]] = 1

        period = time.time() - t1
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period, right_count / total_count))
        for iter in label_dict:
            print("class {3} {4}: eval count:{0}/{1} predict accuracy:{2}".format(label_right_count[iter], label_count[iter], label_right_count[iter] / label_count[iter], iter, label_dict[iter]))

if __name__ == '__main__':
    eval_all()
# qc 1.1
# 0	diuzhen01
# 1	diuzhen02
# 2	diuzhen03
# 3	zhedang01
# 4	zhedang02
# 5	zhedang03
# 6	zhedang04


#2.2
# 0	chaodi
# 1	chaotian
# 2	dropframes
# 3	zhechang
# 4	zhedang

#2.3
# 0	exception
# 1	zhechang

#2.4 yichang1.0
# 0	chaodi
# 1	dropframes
# 2	zhedang

#2.5 yichang1.1
# 0	dropframes
# 1	zhedang