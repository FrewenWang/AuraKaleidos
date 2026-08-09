import os
import onnx
import numpy as np
import onnxruntime as rt
from onnx import shape_inference
import sys

'''
usage:
    python run_onnx_all.py xx.onnx data.raw
'''
axis_order = {
    0: [],
    1: [0],
    2: [0, 1],
    3: [0, 1, 2],
    4: [0, 2, 3, 1],
    5: [0, 1, 2, 3, 4]
}


def get_tensor_shape(tensor):
    dims = tensor.type.tensor_type.shape.dim
    n = len(dims)
    return [dims[i].dim_value for i in range(n)]


def runtime_infer(onnx_model):
    # 获取onnx模型的图结构
    graph = onnx_model.graph
    input_shape = get_tensor_shape(graph.input[0])
    graph.output.insert(0, graph.input[0])

    for i, tensor in enumerate(graph.value_info):
        graph.output.insert(i + 1, tensor)

    model_file = "temp_all_output.onnx"
    onnx.save(onnx_model, model_file)

    #  进行
    sess = rt.InferenceSession(model_file)
    input_name = sess.get_inputs()[0].name
    # input_data = np.ones(input_shape, dtype=np.float32)
    try:
        # assume the data is N * H *W *C, as QNN requirement
        print("Reshape: H:{0},W:{1},C:{2}".format(
            INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL))
        input_data = np.fromfile(
            TEST_INPUT, np.float32).reshape(1, int(INPUT_HEIGHT), int(INPUT_WIDTH), int(INPUT_CHANNEL))
    except:
        print(
            'try to reshape (1,416,416,3) error and change above line to set correct shape')
        raise Exception()

    input_data = np.transpose(input_data, [0, 3, 1, 2])
    outputs = {}

    result_dir = MODEL_DIR + '/Result/'
    if not os.path.exists(result_dir):
        print('Create dir : ' + result_dir)
        os.mkdir(result_dir)

    result_dir = MODEL_DIR + '/Result/OnnxResult/'
    if not os.path.exists(result_dir):
        print('Create dir : ' + result_dir)
        os.mkdir(result_dir)

    for out in sess.get_outputs():
        try:
            tensor = sess.run([out.name], {input_name: input_data})
            result_name = result_dir + str(out.name) + '.raw'
            transpose_shape = axis_order[len(tensor[0].shape)]
            tmp = np.transpose(tensor[0], transpose_shape)
            tmp.astype(np.float32).tofile(result_name)
            outputs[str(out.name)] = np.array(tensor[0]).shape
            print(out.name, 'done!')
        except:
            print('error when store data')
            pass

        # tensor = sess.run([out.name], {input_name: input_data})
        # result_name ='OnnxResult/' + str(out.name) + '.raw'
        # transpose_shape = axis_order[len(tensor[0].shape)]
        # tmp = np.transpose(tensor[0], transpose_shape)
        # tmp.astype(np.float32).tofile(result_name)
        # outputs[str(out.name)] = np.array(tensor[0]).shape
        # print(out.name, 'done!')
    os.remove(model_file)
    return outputs


def infer_shapes(model_file, running_mode=False):
    # 将序列化的 model_file 加载到内存中。
    # onnx.load()是Python的onnx库中的一个函数，用于加载ONNX（开放神经网络交换）模型。
    # ONNX是一种用于表示深度学习模型的开放标准。通过使用ONNX，不同的AI工具之间可以更好地共享模型。
    onnx_model = onnx.load(model_file)
    # 检查模型的一致性。如果检车失败，则会引发异常。
    onnx.checker.check_model(onnx_model)
    # shape_inference.infer_shapes() 是ONNX库中的一个函数，它用于推断ONNX模型图中所有节点的形状。
    # 这个功能对于优化和转换模型非常有用，因为一些转换可能需要知道特定操作的输入和输出形状。
    inferred_onnx_model = shape_inference.infer_shapes(onnx_model)

    # save_path = model_file[:-5] + "_new.onnx"
    # onnx.save(inferred_onnx_model, save_path)
    # print("Model is saved in:", save_path)

    outputs = {}
    if running_mode:
        outputs = runtime_infer(inferred_onnx_model)
    else:
        graph = inferred_onnx_model.graph
        # only 1 input tensor
        tensor = graph.input[0]
        outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # process tensor
        for tensor in graph.value_info:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
        # QnnResult tensor
        for tensor in graph.output:
            outputs[str(tensor.name)] = get_tensor_shape(tensor)
    return outputs


if __name__ == '__main__':
    MODEL_DIR = sys.argv[1]
    MODEL_ONNX = MODEL_DIR + "/" + sys.argv[2]
    TEST_INPUT = MODEL_DIR + "/" + sys.argv[3]
    INPUT_HEIGHT = sys.argv[4]
    INPUT_WIDTH = sys.argv[5]
    INPUT_CHANNEL = sys.argv[6]

    print("MODEL_DIR = {}".format(MODEL_DIR))
    print("MODEL_ONNX = {}".format(MODEL_ONNX))
    print("TEST_INPUT = {}".format(TEST_INPUT))
    print("INPUT_HEIGHT = {}".format(INPUT_HEIGHT))
    print("INPUT_WIDTH = {}".format(INPUT_WIDTH))
    print("INPUT_CHANNEL = {}".format(INPUT_CHANNEL))

    # 执行程序：python exec_onnx.py  ./test.onnx  ./input.raw
    outputs = infer_shapes(MODEL_ONNX, True)
    # outputs = infer_shapes(model_1, False)
    print(outputs)
