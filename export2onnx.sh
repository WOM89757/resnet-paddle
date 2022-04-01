# paddle2onnx  --model_dir model/ --model_filename model.pdmodel --params_filename model.pdparams --opset_version 11 --save_file model/test.onnx

paddle2onnx --model_dir freeze-model-zhedang-2.3.1/  --save_file model/test.onnx --opset_version 10 --enable_onnx_checker True