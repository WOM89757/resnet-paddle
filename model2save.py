import random
import time
import codecs
import sys
import functools
import math
import paddle
import paddle.fluid as fluid

# use_gpu = True
# place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
# exe = fluid.Executor(place)
# save_freeze_dir = "./freeze-model-zhedang-2.3.1"
# paddle.enable_static()
# [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=save_freeze_dir, executor=exe)
# # print(fetch_targets)


# # save by fluid.io.save_params
# model_path = "./freeze-model-zhedang-2.3.1"
# # fluid.io.save_params(exe, model_path)

# # load
# state_dict = paddle.load(model_path)
# # print(state_dict)

# inference_program = fluid.default_main_program()


# paddle.save(inference_program.state_dict(), "model/model.pdparams")
# paddle.save(inference_program, "model/model.pdmodel")

# save by fluid.io.save_params
model_path =  "./freeze-model-zhedang-2.3.1"
# fluid.io.save_params(exe, model_path, filename="__params__")

# load
import os
params_file_path = os.path.join(model_path, "__params__")
var_list = fluid.default_main_program().all_parameters()
state_dict = paddle.io.load_program_state(params_file_path, var_list)
print(state_dict)


