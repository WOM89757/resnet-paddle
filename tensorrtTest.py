import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import torch
import cv2
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad

import sys


import torchvision.transforms as transforms

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

from PIL import Image


# image = Image.open('/home/wangmao/code/datasets/img3.2.1/test/chaodi03/chaodi03_image_267.jpg') # 289
# image = Image.open('/home/wangmao/code/datasets/img3.2.1/train/chaodi03/chaodi03_image_267.jpg') # 289
image = Image.open(sys.argv[1]) # 289
# image.show()
# img = np.array(image)
# img = torch.from_numpy(img)
# img = transforms.Resize([224,244])(image)
# img = transforms.ToTensor()(img)
# img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)

# img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224

# img = transforms.ToTensor()(img)
# img = interpolate(img, size=[3, 224, 224], mode="bilinear", align_corners=None)

img = get_test_transform()(image)
img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
print("plt:{}".format(img.shape))
# print(img)

imageop = cv2.imread('/home/wangmao/code/datasets/img3.2.1/train/chaodi03/chaodi03_image_8.jpg')

imgop = cv2.cvtColor(imageop, cv2.COLOR_BGR2RGB) 
# cv2.imshow("opencv", imageop)

imgop = cv2.resize(imgop, (224, 224), Image.Resampling.BILINEAR)
mean_rgb = [127.5, 127.5, 127.5]
imgop = np.array(imgop).astype('float32')
imgop -= mean_rgb
imgop = imgop.transpose((2, 0, 1))  # HWC to CHW
imgop *= 0.007843
imgop = imgop[np.newaxis,:]
print("opencv:{}".format(imgop.shape))
# print(imgop)

# cv2.waitKey(0)

print("input img mean {} and std {}".format(img.mean(), img.std()))
print("input imgop mean {} and std {}".format(imgop.mean(), imgop.std()))
# 1. 确定batch size大小，与导出的trt模型保持一致
BATCH_SIZE = 1          

# 2. 选择是否采用FP16精度，与导出的trt模型保持一致
USE_FP16 = False                                         
target_dtype = np.float16 if USE_FP16 else np.float32

# 3. 创建Runtime，加载TRT引擎
f = open("./model/resnet50_qc_1.2.1.fp32.trtmodel", "rb")                     # 读取trt模型
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
context = engine.create_execution_context()             # 创建context

# 4. 分配input和output内存
# input_batch = img
input_batch = to_numpy(img)

output = np.empty([BATCH_SIZE, 7], dtype = target_dtype)

d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()

# 5. 创建predict函数
def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output

# 6. 调用predict函数进行推理，并记录推理时间
# def preprocess_input(input):      # input_batch无法直接传给模型，还需要做一定的预处理
#     # 此处可以添加一些其它的预处理操作（如标准化、归一化等）
#     result = torch.from_numpy(input).transpose(0,2).transpose(1,2)   # 利用torch中的transpose,使(224,224,3)——>(3,224,224)
#     return np.array(result, dtype=target_dtype)


preprocessed_inputs = np.array([input for input in input_batch])  # (BATCH_SIZE,224,224,3)——>(BATCH_SIZE,3,224,224)
# preprocessed_inputs = np.array([preprocess_input(input) for input in input_batch])  # (BATCH_SIZE,224,224,3)——>(BATCH_SIZE,3,224,224)
# print("Warming up...")
# pred = predict(preprocessed_inputs)
# print("Done warming up!")

t0 = time.time()
# pred = predict(preprocessed_inputs)
pred = predict(preprocessed_inputs)
# print(pred.shape)
np.set_printoptions(suppress=True)

print(pred)
t = time.time() - t0

print("Prediction: cost {:.4f}s".format(t))
# print("Prediction: {} cost {:.4f}s".format(pred, t))