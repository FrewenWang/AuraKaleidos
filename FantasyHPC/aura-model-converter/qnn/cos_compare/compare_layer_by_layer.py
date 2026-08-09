import json
import numpy as np
from collections import OrderedDict
import os
import sys


def rename_onnx_folder(folder):
    for item in os.listdir(folder):
        item_array = item.split('.')
        item_name = '_'.join(item_array[:-1])
        item_name = item_name + '.raw'
        if item == item_name:
            continue
        os.system('cp {} {}'.format(os.path.join(
            folder, item), os.path.join(folder, item_name)))
        # cmd = 'cp {} {}'.format(os.path.join(
        #     folder, item), os.path.join(folder, item_name))
        # print(cmd)
        # if item_name == 'relu_1_tmp_0_tmp_quantized_dequantized_0.raw':
        #    print('bingo')
        #    assert 0


# assert 0

def cosine_sim(data1, data2):
    data1, data2 = data1.flatten(), data2.flatten()
    num = data1.dot(data2.T)
    denom = np.linalg.norm(data1) * np.linalg.norm(data2)
    return num / denom


def get_layers_names(NET_JSON_FILE):
    fp = open(NET_JSON_FILE)
    items = json.load(fp)
    graph = items['graph']
    tensors = graph['tensors']

    layer_dict = OrderedDict()
    key_list = []
    for key in tensors.keys():
        tensor_item = tensors[key]
        if tensor_item["type"] in [1, 3]:
            layer_dict[key] = tensor_item
            key_list.append(key)

    return key_list, layer_dict


NET_JSON_FILE = sys.argv[1]
RESULT_DIR = sys.argv[2]
RESULT_ONNX = RESULT_DIR + '/OnnxResult/'
RESULT_QNN = RESULT_DIR + '/QnnResult/Result_0/'


def analysis():
    print("Get Layer Names From Json File : \n>> {}".format(NET_JSON_FILE))
    names, _ = get_layers_names(NET_JSON_FILE)
    for name in names:
        try:
            name = name + '.raw'
            data1 = np.fromfile(os.path.join(RESULT_QNN, name), np.float32)
            data2 = np.fromfile(os.path.join(RESULT_ONNX, name), np.float32)
            cons = cosine_sim(data1, data2)
            print('Cosine : {} -> {}'.format(cons, name))
        except IOError:
            print("ERROR : No Such RawData [ {} ] !!!".format(name))


if __name__ == '__main__':
    print("-- NET JSON FILE   = " + NET_JSON_FILE)
    print("-- RESULT ONNX DIR = " + RESULT_ONNX)
    print("-- RESULT QNN DIR  = " + RESULT_QNN)

    rename_onnx_folder(RESULT_ONNX)
    analysis()
