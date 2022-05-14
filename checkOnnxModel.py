import onnx
import numpy as np
import onnxruntime
import sys
import cv2
import torch
from PIL import Image


onnx_file = './model/resnet50_qc_1.2.1.onnx'
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

# x = np.random.random((1,3,224,224)).astype('float32')
# # print("x:",x)

# # predict by ONNX Runtime
# ort_sess = onnxruntime.InferenceSession(onnx_file)
# ort_inputs = {ort_sess.get_inputs()[0].name: x}
# ort_outs = ort_sess.run(None, ort_inputs)

import torchvision.transforms as transforms

means = [0.5, 0.5, 0.5]
stds = [0.5, 0.5, 0.5]
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def cv2_transform(cv2_img):
    img = cv2_img.copy()
    img = cv2.resize(cv2_img, (224, 224), Image.BILINEAR) # 如果注释这两行，预测结果是一样的
    img = np.array(img[:, :, ::-1], dtype=np.float32)
    img = img / 255
    img = img - np.array(means, dtype=np.float32)
    img = img / np.array(stds, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis,:]
    img = torch.from_numpy(img)
    return img

image = Image.open(sys.argv[1]) # 289
# 0.50000875
img = get_test_transform()(image)
img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
print("input img mean {} and std {}".format(img.mean(), img.std()))
# print(img.shape)


imageop = cv2.imread(sys.argv[1])
imageop = cv2_transform(imageop)
print("input imageop mean {} and std {}".format(imageop.mean(), imageop.std()))
# print(imageop.shape)

##onnx测试
resnet_session = onnxruntime.InferenceSession("./model/resnet50_qc_1.3.2.onnx")
# resnet_session = onnxruntime.InferenceSession("./model/resnet50_qc_1.2.1.onnx")
#compute ONNX Runtime output prediction
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]
np.set_printoptions(suppress=True)
print("onnx weights {},onnx prediction:{}".format(np.around(outs, 4), outs.argmax(axis=1)[0]))

inputs = {resnet_session.get_inputs()[0].name: to_numpy(imageop)}
outs = resnet_session.run(None, inputs)[0]
print("onnx weights {},onnx prediction:{}".format(np.around(outs, 4), outs.argmax(axis=1)[0]))

print("Exported model has been predicted by ONNXRuntime!")
