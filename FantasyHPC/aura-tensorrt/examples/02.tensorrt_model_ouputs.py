import tensorrt as trt

# 加载TensorRT引擎
logger = trt.Logger(trt.Logger.INFO)
with open('./model.trt', "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
for idx in range(engine.num_bindings):
    name = engine.get_tensor_name(idx)
    is_input = engine.get_tensor_mode(name)
    op_type = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print('input id: ', idx, '\tis input: ', is_input, '\tbinding name: ', name, '\tshape: ', shape, '\ttype: ',
          op_type)
