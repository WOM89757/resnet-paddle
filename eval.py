from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label

import os
from re import I
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

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sn
from utils import custom_image_reader

target_size = [3, 224, 224]
mean_rgb = [127.5, 127.5, 127.5]
data_dir = "../datasets/img4.0/"
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
save_freeze_dir = "freeze-model-qc-1.3.2"
# save_freeze_dir = "./freeze-model-qc-1.2.1"
# save_freeze_dir = "./freeze-model-yichang-1.1"
# save_freeze_dir = "./freeze-model-zhedang-2.3"
save_confusion_dir = "./figure_result/" + save_freeze_dir +'/'
if not os.path.exists(save_confusion_dir):
    os.makedirs(save_confusion_dir)
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
    img = resize_img(img, target_size)
    img = np.array(img).astype('float32')
    img -= mean_rgb
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    img = img[np.newaxis,:]
    return img

def eval_batch(val_file_list, data_dir, mode='eval'):
    # val_acc = []
    # val_loss = []
    # test_program = fluid.default_main_program().clone(for_test=True)
    # img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    # label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    # #     main_program = fluid.default_main_program()
    # # exe.run(fluid.default_startup_program())
    # # vaildation
    # val_batch_reader = paddle.batch(custom_image_reader(val_file_list, data_dir, mode, train_parameters),
    #                         batch_size=train_parameters['train_batch_size'],
    #                         drop_last=True)
    # for step_id, data in enumerate(val_batch_reader()):
    #     loss, acc1, pred_ot = exe.run(test_program,
    #                                     feed=feeder.feed(data),
    #                                     fetch_list=fetch_targets)
    #     loss = np.mean(np.array(loss))
    #     acc1 = np.mean(np.array(acc1))
    #     val_acc.append(acc1)
    #     val_loss.append(loss)

    # val_loss = np.mean(np.array(val_loss))
    # val_acc = np.mean(np.array(val_acc))
    return 1

def infer(image_path):
    tensor_img = read_image(image_path)
    label = exe.run(inference_program, feed={feed_target_names[0]: tensor_img}, fetch_list=fetch_targets)
    return np.argmax(label), label[0][0]

def auc(actual, pred):
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

def roc_plot(fpr, tpr, title_name, save_figure=False , save_dir = './'):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1],[0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_name)
    plt.legend(loc="lower right")
    if save_figure:
        plt.savefig('{}{}.png'.format(save_dir, title_name))
    else:
        plt.show()

def eval_all(eval_file = "eval.txt", label_file = "label_list.txt"):
    confusion_flag = True
    single_class = False
    class_detail_flag = True
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

    y_true = []
    y_probability = []
    y_predict = []
    with codecs.open(eval_file_path, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        t1 = time.time()

        for line in lines:
            total_count += 1
            parts = line.strip().split()
            result, probability = infer(parts[0])
            y_true.append(int(parts[1]))
            if single_class:
                y_probability.append(probability[np.argmax(probability)])
            else:
                y_probability.append(probability.tolist())
            y_predict.append(result)
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
        acc = right_count / total_count
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period, acc))
        for iter in label_dict:
            print("class {3} {4}: eval count:{0}/{1} predict accuracy:{2}".format(label_right_count[iter], label_count[iter], label_right_count[iter] / label_count[iter], iter, label_dict[iter]))
    
    if confusion_flag:
        matrixes = metrics.confusion_matrix(y_true, y_predict)
        print(matrixes)
        con_mat = matrixes
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        confusion_name = 'Confusion_Matrix—{}'.format(save_freeze_dir[save_freeze_dir.rfind('/')+1:])
        plt.figure(confusion_name, figsize=(7, 7))
        sn.heatmap(con_mat_norm, annot=True, cmap='Blues')
        # sn.heatmap(con_mat_norm, annot=True, fmt='.20g', cmap='Blues')
        plt.ylim(0, 7)
        plt.xlabel('Predicted labels')
        plt.ylabel('True      labels')
        plt.title(confusion_name)
        plt.savefig('{}{}.png'.format(save_confusion_dir, confusion_name))
        
        target_names = []
        for iter in label_dict:
            target_names.append(label_dict[iter])
        print(metrics.classification_report(y_true, y_predict, target_names=target_names))

        if not single_class:
            label_types = np.unique(y_true)
            n_class = label_types.size
            y_one_hot = label_binarize(y_true, classes=np.arange(n_class))
            y_one_hot = np.array(y_one_hot)
            y_n_probability = np.array(y_probability)
            y_true = y_one_hot
            y_probability = y_n_probability
            if class_detail_flag:
                for i in range(len(label_dict)):
                    print('class:{} auc: {}'.format(i, auc(y_true[:,i], y_probability[:,i])))
                    fpr, tpr, thresholds = metrics.roc_curve(y_true[:,i], y_probability[:,i], pos_label=1)
                    Roc_name = 'Roc_Curve_class-{}—{}'.format(i, save_freeze_dir[save_freeze_dir.rfind('/')+1:])
                    roc_plot(fpr, tpr, title_name = Roc_name, save_figure=True, save_dir=save_confusion_dir)
            y_true = y_one_hot.ravel()
            y_probability = y_n_probability.ravel()


        print('auc: {}'.format(auc(y_true, y_probability)))
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probability, pos_label=1)
        # print('fpr: {}'.format(fpr))
        # print('tpr: {}'.format(tpr))
        # print('thresholds: {}'.format(thresholds))
        Roc_name = 'Roc_Curve—{}'.format(save_freeze_dir[save_freeze_dir.rfind('/')+1:])
        roc_plot(fpr, tpr, title_name = Roc_name, save_figure=True, save_dir=save_confusion_dir)

    return acc
if __name__ == '__main__':
    eval_all()
    # print('Macro f1-score', metrics.f1_score(y_true, y_predict, average='macro'))  

    # y_probability = [0.99927396, 0.9002646, 0.859398, 0.74960023, 0.9975351, 0.95004493, 0.6334533, 0.8696714, 0.9960569, 0.9883613, 0.96473897, 0.9985196, 0.98960376, 0.999926, 0.61607003, 0.9450607, 0.98049164, 0.9204282, 0.9968099, 0.97808456, 0.99651235, 0.9744071, 0.52571464, 0.8292975, 0.83812636, 0.9877457, 0.9958954, 0.9987515, 0.98882025, 0.9995351, 0.97865903, 0.9703743, 0.9938699, 0.9996371, 0.950276, 0.9964144, 0.83568335, 0.99981445, 0.98095703, 0.9975243, 0.9572031, 0.8601701, 0.5613634, 0.832151, 0.94766474, 0.9370868, 0.99978215, 0.99093914, 0.96831256, 0.9902963, 0.9058853, 0.9946549, 0.67945254, 0.9961055, 0.99864846, 0.97535354, 0.9918006, 0.99185234, 0.9931062, 0.9978083, 0.95421845, 0.99768066, 0.9573395, 0.9985832, 0.99112076, 0.9904232, 0.8757731, 0.8696715, 0.9996024, 0.9900865, 0.99865556, 0.9915036, 0.99393207, 0.937378, 0.9987754, 0.9503579, 0.9825326, 0.9803798, 0.82532984, 0.9989524, 0.9987207, 0.9928948, 0.65259624, 0.59944427, 0.64907736, 0.9988657, 0.998749, 0.9493766, 0.9840211, 0.6020472, 0.97871494, 0.9883758, 0.84306264, 0.9984187, 0.9997638, 0.96001476, 0.9958341, 0.9518156, 0.9962794, 0.9981902, 0.9115795, 0.997424, 0.8750451, 0.6757215, 0.99896884, 0.7785157, 0.99445057, 0.8914959, 0.9916739, 0.9986094, 0.99172133, 0.83821905, 0.9485884, 0.99929225, 0.9995198, 0.96360815, 0.9343848, 0.9769644, 0.93226004, 0.98636997, 0.8607898, 0.5550513, 0.83927196, 0.950513, 0.9959716, 0.99661237, 0.98595667, 0.9228699, 0.9972512, 0.9953434, 0.95133775, 0.9872931, 0.9984484, 0.92535067, 0.5763143, 0.65055126, 0.9974095, 0.99166733, 0.888224, 0.9987464, 0.95707387, 0.9782451, 0.9709853, 0.83916473, 0.87457913, 0.99848217, 0.8594213, 0.8501661, 0.9908942, 0.9681571, 0.5442134, 0.7600526, 0.64934, 0.9435101, 0.99430096, 0.596937, 0.97444916, 0.97126883, 0.7289165, 0.97044486, 0.97241366, 0.9653675, 0.9709608, 0.97182053, 0.70845443, 0.9724992, 0.97334814, 0.97239923, 0.9158304, 0.9591318, 0.9775718, 0.9688938, 0.9658395, 0.88832206, 0.97939277, 0.91214323, 0.99983644, 0.9974383, 0.98749626, 0.5494531, 0.9746714, 0.99684143, 0.9932106, 0.97807306, 0.97975177, 0.9768389, 0.731674, 0.99376136, 0.973409, 0.9976519, 0.9978859, 0.94718665, 0.7739086, 0.99376833, 0.9209907, 0.94539094, 0.77283555, 0.985261, 0.9258699, 0.696563, 0.7648795, 0.9906495, 0.8493987, 0.7919357, 0.6152166, 0.87269276, 0.8783416, 0.9948027, 0.995732, 0.9532236, 0.97884345, 0.9974752, 0.95460135, 0.709318, 0.75281775, 0.9804331, 0.9791005, 0.99801993, 0.95047164, 0.9712683, 0.8566027, 0.98456454, 0.97877866, 0.80862916, 0.9708594, 0.95345104, 0.90013623, 0.7575288, 0.9837869, 0.9983038, 0.93206954, 0.9829202, 0.99126464, 0.9987972, 0.9848124, 0.9681248, 0.68529505, 0.98642564, 0.9831751, 0.9045052, 0.99673957, 0.99585766, 0.99054354, 0.9932146, 0.7535829, 0.5637781, 0.8679308, 0.85415924, 0.9799147, 0.9974873, 0.9255573, 0.6912083, 0.8474625, 0.98133826, 0.94498956, 0.9806911, 0.8786007, 0.9828082, 0.942926, 0.9907979, 0.49660555, 0.98085403, 0.99591357, 0.97399604, 0.9895773, 0.99763024, 0.97042346, 0.9971046, 0.99614704, 0.96734893, 0.64611965, 0.9587956, 0.77855515, 0.7852879, 0.98952883, 0.9776479, 0.9595125, 0.9759772, 0.9656655, 0.98541623, 0.97796625, 0.9910602, 0.9519986, 0.96476525, 0.93442804, 0.97821236, 0.9940299, 0.9983974, 0.7112093, 0.91737235, 0.9862693, 0.8940774, 0.94271386, 0.9874735, 0.98875624, 0.6379392, 0.9943963, 0.49042588, 0.9930831, 0.9912395, 0.6988801, 0.8182191, 0.5699913, 0.99653083, 0.9853888, 0.9939883, 0.89905494, 0.9763139, 0.8204537, 0.77656955, 0.98685557, 0.89763093, 0.98285633, 0.9938689, 0.7709143, 0.73626447, 0.9925746, 0.5051197, 0.96426624, 0.9854034, 0.77084005, 0.9291067, 0.67518216, 0.57356495, 0.99041384, 0.98167694, 0.7732671, 0.9949195, 0.9939049, 0.9819089, 0.8861418, 0.9930099, 0.9014066, 0.9786038, 0.9505871, 0.94158626, 0.9527099, 0.9738488, 0.9988199, 0.5868899, 0.97583276, 0.9883692, 0.9769782, 0.98630226, 0.9976108, 0.9660818, 0.80665714, 0.99268407, 0.9152931, 0.89049196, 0.997997, 0.9890171, 0.9455881, 0.99024206, 0.99766266, 0.9040923, 0.97130764, 0.7870597, 0.7239654, 0.9530727, 0.977372, 0.85014, 0.99225473, 0.9296414, 0.6401556, 0.8726767, 0.8761295, 0.9914484, 0.9689679, 0.98103297, 0.96764004, 0.57053006, 0.86618745, 0.9424575, 0.98880416, 0.9676599, 0.9747635, 0.9912348, 0.98625654, 0.6488722, 0.992804, 0.9693971, 0.8827902, 0.9962424, 0.98582804, 0.967172, 0.9869997, 0.9927971, 0.96014386, 0.99608254, 0.9890723, 0.9844689, 0.98744315, 0.9435654, 0.98487735, 0.90023255, 0.9751185, 0.99804604, 0.93887323, 0.8984246, 0.73173714, 0.91897047, 0.90687007, 0.9725424, 0.95087296, 0.9948598, 0.7910798, 0.99451506, 0.6749202, 0.6237281, 0.896391, 0.9444292, 0.9719812, 0.9331556, 0.98223346, 0.86391985, 0.9858466, 0.98993427, 0.9723478, 0.7226662, 0.9892958, 0.79119235, 0.82656854, 0.8405154, 0.9972588, 0.8123891, 0.99416554, 0.99835473, 0.929484, 0.9136867, 0.95576715, 0.78376156, 0.97999716, 0.99701273, 0.789886, 0.9288886, 0.9970969, 0.93758124, 0.99487066, 0.9791118, 0.98957133, 0.97542, 0.9768926, 0.99080336, 0.99676114, 0.61113095, 0.9960747, 0.92860913, 0.5817461, 0.9754667, 0.9781804, 0.9026331, 0.971882, 0.97030836, 0.9715712, 0.9904364, 0.80830574, 0.9846256, 0.9907302, 0.75632346, 0.8338001, 0.7240022, 0.9892161, 0.9888842, 0.61401623, 0.98747975, 0.5308432, 0.89301956, 0.72132456, 0.9866766, 0.97476465, 0.89815193, 0.9923275, 0.51349586, 0.96805286, 0.98419064, 0.9915324, 0.8975483, 0.75969434, 0.7140637, 0.9344573, 0.9869081, 0.82663876, 0.9660461, 0.99915874, 0.78217393, 0.89829326, 0.93778336, 0.9200574, 0.9857497, 0.52427113, 0.99642354, 0.82935876, 0.5315492, 0.9601905, 0.84000665, 0.8968394, 0.5775102, 0.81661016, 0.9263597, 0.9155618, 0.9627356, 0.9629961, 0.66403687, 0.5232462, 0.9309252, 0.8934058, 0.8943284, 0.7312265, 0.9453816, 0.8586237, 0.989884, 0.641062, 0.99134845, 0.9932594, 0.6432614, 0.9677839, 0.83908045, 0.89998674, 0.9747205, 0.98113143, 0.9262831, 0.97285426, 0.85548735, 0.97021526, 0.5140165, 0.899369, 0.8240333, 0.8444441, 0.90135777, 0.9732963, 0.988332, 0.82611346, 0.9184093, 0.9891684, 0.9858298, 0.9968267, 0.8933453, 0.9826502, 0.87359667, 0.96708375, 0.93801785, 0.9604122, 0.9928884, 0.99022, 0.76018226, 0.7207765, 0.9826087, 0.98393905, 0.99774593, 0.985688, 0.61556625, 0.98538, 0.9688909, 0.5220259, 0.51365083, 0.7193301, 0.8027259, 0.6650625, 0.69521564, 0.9539374, 0.6705705, 0.93674105, 0.9035991, 0.9853581, 0.7264825, 0.9522612, 0.99809736, 0.9784801, 0.94899476, 0.97122264, 0.55880207, 0.8786627, 0.96786666, 0.6452147, 0.9921631, 0.9919871, 0.7989179, 0.99073213, 0.98396665, 0.9932876, 0.9692769, 0.92139626, 0.95903784, 0.9209915, 0.6437766, 0.99194396, 0.9630983, 0.9939818, 0.9254553, 0.68181664, 0.99111605, 0.9323566, 0.9694016, 0.98026854, 0.88092655, 0.727577, 0.72564006, 0.5640022, 0.9501048, 0.687757, 0.99618834, 0.54326713, 0.9872069, 0.43973613, 0.99266195, 0.7279154, 0.6574699, 0.7852248, 0.64199525, 0.5786506, 0.607011, 0.6014856, 0.7442646, 0.70802784, 0.766533, 0.9959611, 0.9957267, 0.6491713, 0.7143341, 0.66117245, 0.81310725, 0.9576554, 0.67270845, 0.826294, 0.99687046, 0.5098929, 0.4915688, 0.9963406, 0.4519897, 0.71886, 0.9924115, 0.99121255, 0.64880705, 0.9965373, 0.73202026, 0.68559456, 0.6109672, 0.6234739, 0.996442, 0.82301813, 0.5158471, 0.6236604, 0.6348981, 0.99651414, 0.99476975, 0.55941385, 0.99500257, 0.5478193, 0.99582386, 0.9897685, 0.67626464, 0.5352626, 0.81853384, 0.8098164, 0.9957736, 0.9965005, 0.45735666, 0.7285645, 0.98057246, 0.9957712, 0.9959825, 0.644318, 0.6441447, 0.72336215, 0.8450202, 0.7943702, 0.996316, 0.7710484, 0.701765, 0.995862, 0.9963379, 0.8703297, 0.9893511, 0.643547, 0.5483065, 0.9900169, 0.6669907, 0.6314696, 0.6840218, 0.66042113, 0.69907105, 0.5651364, 0.9274415, 0.9879753, 0.57989883, 0.51505476, 0.63519514, 0.9952494, 0.7627842, 0.99108785, 0.6250354, 0.9956246, 0.9955721, 0.5096821, 0.9891671, 0.99530655, 0.6045272, 0.9948278, 0.88143694, 0.9964761, 0.50620514, 0.99438584, 0.5337295, 0.9814455, 0.99511945, 0.55012643, 0.99915993, 0.96569544, 0.9944857, 0.99931955, 0.9675445, 0.9029325, 0.99951434, 0.99572265, 0.9692199, 0.9762374, 0.5763236, 0.99981385, 0.99996185, 0.9675445, 0.9998204, 0.982429, 0.9990897, 0.9977822, 0.9086463, 0.5465319, 0.99473757, 0.9312493, 0.9831182, 0.9997727, 0.99809366, 0.9984579, 0.998801, 0.9996463, 0.9998623, 0.9997459, 0.9999639, 0.9996525, 0.99980515, 0.8729008, 0.97496945, 0.9928052, 0.92664903, 0.99962044]
    # y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  
    # y_predict = [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 6, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 3, 4, 3, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 0, 4, 4, 4, 3, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 0, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 3, 4, 4, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6]
    # matrixes = sm.confusion_matrix(y_true, y_predict)
    # print(matrixes)
    # con_mat = matrixes
    # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]     # 归一化
    # con_mat_norm = np.around(con_mat_norm, decimals=2)
    # mp.figure('Confusion Matrix', figsize=(7, 7))
    # sn.heatmap(con_mat_norm, annot=True, cmap='Blues')
    # # sn.heatmap(con_mat_norm, annot=True, fmt='.20g', cmap='Blues')
    # mp.ylim(0, 7)
    # mp.xlabel('Predicted labels')
    # mp.ylabel('True      labels')
    # mp.savefig('./1.png')


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