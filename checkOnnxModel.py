import onnx
import numpy as np
import onnxruntime

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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

image = Image.open('/home/wangmao/code/datasets/img3.2.1/test/chaodi03/chaodi03_image_267.jpg') # 289
# 0.50000875
img = get_test_transform()(image)
img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
print("input img mean {} and std {}".format(img.mean(), img.std()))
##onnx测试
resnet_session = onnxruntime.InferenceSession("./model/resnet50_qc_1.2.1.onnx")
#compute ONNX Runtime output prediction
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]
np.set_printoptions(suppress=True)
print("onnx weights", outs)
print("onnx prediction", outs.argmax(axis=1)[0])

print("Exported model has been predicted by ONNXRuntime!")
