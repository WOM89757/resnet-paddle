import codecs
from operator import contains
import os
import random
import shutil
from time import process_time_ns
from PIL import Image

train_ratio = 4.0 / 5

dir_class = '../datasets/img2.2/'

train_image_dir = os.path.join(dir_class, "trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)
    
eval_image_dir = os.path.join(dir_class, "evalImageSet")
if not os.path.exists(eval_image_dir):
    os.makedirs(eval_image_dir)

train_file = codecs.open(os.path.join(dir_class, "train.txt"), 'w')
eval_file = codecs.open(os.path.join(dir_class, "eval.txt"), 'w')

for name_dir in os.listdir(dir_class):
    # print(name_dir)
    # if name_dir.find('.txt'): 
    if 'txt' in name_dir or 'Set' in name_dir: 
        continue
    all_file_dir = dir_class + name_dir
    print(all_file_dir)

    class_list = [c for c in os.listdir(all_file_dir) if os.path.isdir(os.path.join(all_file_dir, c)) and not c.endswith('Set') and not c.startswith('.') and not c.endswith('txt')]
    class_list.sort()
    print(class_list)

    with codecs.open(os.path.join(dir_class, "label_list.txt"), "w") as label_list:
        label_id = 0
        for class_dir in class_list:
            label_list.write("{0}\t{1}\n".format(label_id, class_dir))
            # print(str(class_dir) + '--' + str(label_id))
            image_path_pre = os.path.join(all_file_dir, class_dir)
            for file in os.listdir(image_path_pre):
                # print(file)
                try:
                    img = Image.open(os.path.join(image_path_pre, file))
                    # if random.uniform(0, 1) <= train_ratio:
                    if 'train' in name_dir:
                        shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(train_image_dir, file))
                        train_file.write("{0}\t{1}\n".format(os.path.join(train_image_dir, file), label_id))
                    else:
                        shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(eval_image_dir, file))
                        eval_file.write("{0}\t{1}\n".format(os.path.join(eval_image_dir, file), label_id))
                except Exception as e:
                    pass
                    # 存在一些文件打不开，此处需要稍作清洗
            label_id += 1
                



train_file.close()
eval_file.close()

print("finished split!!!")