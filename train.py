# -*- coding: UTF-8 -*-
"""
训练常用视觉基础网络，用于分类任务
需要将训练图片，类别文件 label_list.txt 放置在同一个文件夹下
程序会先读取 train.txt 文件获取类别数和图片数量
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
from resNet50 import ResNet
from visualdl import LogWriter

from utils import *
from eval import eval_all


train_parameters = {
    "input_size": [3, 224, 224],
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得
    "label_dict": {},
    "data_dir": "../datasets/img3.2.1/",  # 训练数据存储地址
    "train_file_list": "train.txt",
    "label_file": "label_list.txt",
    "save_freeze_dir": "./freeze-model-qc-1.2.1",
    "save_persistable_dir": "./persistable-params-qc-1.2.1",
    "continue_train": True,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型
    "pretrained": True,            # 是否使用预训练的模型
    "pretrained_dir": "./ResNet50_pretrained",
    "mode": "train",
    "num_epochs": 100,
    "train_batch_size": 24,
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "use_gpu": True,
    "image_enhance_strategy": {  # 图像增强相关策略
        "need_distort": True,  # 是否启用图像颜色增强
        "need_rotate": True,   # 是否需要增加随机角度
        "need_crop": True,      # 是否要增加裁剪
        "need_flip": True,      # 是否要增加水平随机翻转
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 3,
        "good_acc1": 0.96
    },
    "rsm_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "momentum_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "sgd_strategy": {
        "learning_rate": 0.002,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "adam_strategy": {
        "learning_rate": 0.002
    }
}


def init_train_parameters():
    """
    :return:
    """
    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)

def train(logger):
    train_prog = fluid.Program()
    train_startup = fluid.Program()
    logger.info("create prog success")
    logger.info("train config: %s", str(train_parameters))
    logger.info("build input custom reader and data feeder")
    file_list = os.path.join(train_parameters['data_dir'], "train.txt")
    mode = train_parameters['mode']
    batch_reader = paddle.batch(custom_image_reader(file_list, train_parameters['data_dir'], mode, train_parameters),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=True)
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    paddle.enable_static()
    
    img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    logger.info("build newwork")
    model = ResNet()
    out = model.net(input=img, class_dim=train_parameters['class_dim'])
    cost = fluid.layers.cross_entropy(out, label)
    # print(out)
    # print(label)
    # cost = fluid.layers.sigmoid_cross_entropy_with_logits(out, label, ignore_index=-1, normalize=True)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    
    # optimizer = optimizer_rms_setting(train_parameters)
    # optimizer = optimizer_momentum_setting(train_parameters)
    # optimizer = optimizer_sgd_setting(train_parameters)
    optimizer = optimizer_adam_setting(train_parameters)
    optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    main_program = fluid.default_main_program()
    exe.run(fluid.default_startup_program())
    train_fetch_list = [avg_cost.name, acc_top1.name, out.name]
    
    load_params(exe, main_program, train_parameters)

    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    good_acc1 = stop_strategy['good_acc1']
    successive_count = 0
    stop_train = False
    total_batch_count = 0
    best_acc = 0.95

    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        epoch_acc = []
        epoch_loss = []
        for step_id, data in enumerate(batch_reader()):
            t1 = time.time()
            loss, acc1, pred_ot = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=train_fetch_list)
            # label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
            
            t2 = time.time()
            batch_id += 1
            total_batch_count += 1
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            epoch_acc.append(acc1)
            epoch_loss.append(loss)

            # print(batch_id, '-', step_id)
            if batch_id % 10 == 0:
                logger.info("Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, time {4}".format(pass_id, batch_id, loss, acc1,
                                                                                            "%2.2f sec" % period))
                    
            if acc1 >= good_acc1:
                successive_count += 1
                logger.info("current acc1 {0} meets good {1}, successive count {2}".format(acc1, good_acc1, successive_count))
                fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program,
                                              executor=exe)
                if successive_count >= successive_limit:
                    logger.info('start eval all test data:')
                    eval_acc = eval_all()
                    logger.info('end  eval all test data, eval acc is {}'.format(eval_acc))
                    if eval_acc > 0.8:
                        logger.info("end training")
                        stop_train = True
                        break
                    else:
                        successive_count = 0
                    # logger.info("end training")
                    # stop_train = True
                    # break
            else:
                successive_count = 0

            if total_batch_count % sample_freq == 0:
                logger.info("temp save {0} batch train result, current acc1 {1}".format(total_batch_count, acc1))
                fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
            if acc1 > best_acc:
                best_acc = acc1
                logger.info("best save {0} batch train result, current acc1 {1}".format(total_batch_count, acc1))
                best_model_name = '{}-best-acc-{}-loss-{}'.format(train_parameters['save_persistable_dir'], acc1, loss)
                print(best_model_name)
                # fluid.io.save_persistables(dirname=best_model_name,
                #                            main_program=main_program,
                #                            executor=exe)
                # fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                #                             feeded_var_names=['img'],
                #                             target_vars=[out],
                #                             main_program=main_program,
                #                             executor=exe)


        epoch_loss = np.mean(np.array(epoch_loss))
        epoch_acc = np.mean(np.array(epoch_acc))
        with LogWriter(logdir="./log/" + train_parameters['save_freeze_dir'].rsplit("/", 1)[1]) as writer:
            writer.add_scalar(tag="epoch_acc", step=pass_id, value=epoch_acc)
            writer.add_scalar(tag="epoch_loss", step=pass_id, value=epoch_loss)
        if stop_train:
            break

    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
    fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program,
                                              executor=exe)

