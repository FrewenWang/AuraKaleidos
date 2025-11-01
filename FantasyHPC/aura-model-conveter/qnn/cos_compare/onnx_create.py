import onnx
import onnx.helper as helper
from onnx import numpy_helper

from onnxsim import simplify

import json
from collections import OrderedDict

import struct
import sys
#https://leimao.github.io/blog/ONNX-Python-API/

purne_weights_qat_node = True


def print_tensor_data(initializer: onnx.TensorProto) -> None:
    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        return initializer.float_data
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        print(initializer.int32_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        print(initializer.int64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        print(initializer.double_data)
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        print(initializer.uint64_data)
    elif initializer.data_type == onnx.TensorProto.DataType.INT8:
        print('int8')
        return 0
        #return initializer.int8_data
    else:
        raise NotImplementedError


def make_dummy_node(node ):
    model_input_name = node.name
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                       onnx.TensorProto.FLOAT,
                                       [None, 3, 32, 32])
    

class EncodingHelper():
    def __init__(self,nodes_map = [] ,encoding_name = 'aimet_encoding_debug.json'):        
        self.order_dict = []
        self.weights_order_dict = []
        self.output_name = encoding_name
        
        self.nodes = nodes_map
    
    def __optimizate_encoding(self,node,scale,offset):
        adjust_offset = offset
        is_symmetric = True

        search_node  = None
        for item in self.nodes:
            if node.input[0] in item.output:
                search_node = item
                break
        #for relu, do not use  symmetric       
        if  search_node:       
            if search_node.op_type == 'Relu':
                adjust_offset = 0
                is_symmetric = False
                #better accuracy,symmetric quantization, 0.5
                scale = scale/2
        return scale,offset, is_symmetric
        
    def onnx_quant_encoding_helper(self,node,initializer,is_act_encoding = True):
        if len(node.input) < 3:
            return 
            
        name = node.input[0]
        #https://github.com/onnx/onnx/issues/2825
        if len(initializer[node.input[1]].float_data) < 1:
            raw_data = initializer[node.input[1]].raw_data
            scale = struct.unpack("f", raw_data)[0]
        else:    
            scale =   initializer[node.input[1]].float_data[0]
        offset =  0
        
        if is_act_encoding:
            scale,offset,symmetric = self.__optimizate_encoding(node,scale,offset)
            self.add_act_encoding(name,scale,offset,is_symmetric=symmetric)
        else:
            self.add_weights_encoding(name,scale,offset)
            
            
    def add_act_encoding(self,name,  scale,offset, bw = 8, is_symmetric = True):

        self.order_dict.append([name,scale,offset, bw,is_symmetric])
    
    def add_weights_encoding(self,name,  scale,offset, bw = 8, is_symmetric = True):
        self.weights_order_dict.append([name,scale,offset, bw,is_symmetric])
        
    def transfer_2_aimet_format(self,items):
        
        aimet_formats = []
        for item in items:
            item_dict = {
                item[0] : {
                    'bw': item[3],
                    'offset': item[2],
                    'scale':  item[1],
                    'is_symmetric':item[4]
                }
            }
            aimet_formats.append(item_dict)
        return aimet_formats
    
    def write_to_aimet_format(self):
    # act_encoding format: list[dict{output_name:encoding dict}]
    # param_encoding format: list[dict {name: encoding dict}]
    #aimet format: dict{name:list[dict{encoding},dict{encoding},xxx]}, the encoding dict:
    # "bitwidth":xx,"max": xxx,."min": xxx,"offset": xx,"scale": xxx
        act_encooding = self.transfer_2_aimet_format(self.order_dict)
        param_encoding = self.transfer_2_aimet_format(self.weights_order_dict)
        
        aimet_act_json_file = OrderedDict()
        for item in act_encooding:
           
            for name,info in item.items():
                aimet_format= OrderedDict()
                aimet_format["bitwidth"] = info['bw']
                aimet_format["min"] = info['scale'] * -128
                aimet_format["max"] = info['scale']* 127

                aimet_format["offset"] = info['offset']
                aimet_format["scale"] = info['scale']
                if info['is_symmetric']:
                    aimet_format["is_symmetric"] = "true"
                else:
                    aimet_format["is_symmetric"] = "false"
    
                aimet_act_json_file[name] =[aimet_format]
        #print(aimet_act_json_file)    
        aimet_param_json_file = OrderedDict()
        for item in param_encoding:
            for name,info in item.items():
                aimet_format= OrderedDict()
                aimet_format["bitwidth"] = info['bw']
                aimet_format["min"] = info['scale'] * -128
                aimet_format["max"] = info['scale']* 127
                aimet_format["offset"] = info['offset']
                aimet_format["scale"] = info['scale']
                if info['is_symmetric']:
                    aimet_format["is_symmetric"] = "true"
                else:
                    aimet_format["is_symmetric"] = "false"
                
                aimet_param_json_file[name] = [aimet_format]
        aimet_json_file={}     
        aimet_json_file['activation_encodings'] = aimet_act_json_file
        aimet_json_file['param_encodings']      = aimet_param_json_file
        
        with open(self.output_name,'w') as json_write:
            json.dump(aimet_json_file,json_write,indent=4)

if len(sys.argv) < 2:
    print(' ' * 100)
    print('usage: python onnx_create.py example.onnx')
    print(' ' * 100)
    assert 0 
    
model_name = sys.argv[1]
        
model = onnx.load(model_name)
onnx.checker.check_model(model)

graph_def = model.graph
nodes = graph_def.node

quantizeLinear_nodes=[]
dequantizeLinear_nodes=[]
initializer_map= {}

initializers = graph_def.initializer

for initializer in initializers:
    #print(initializer.name)
    initializer_map[initializer.name] = initializer

#assert 0


for node in nodes:
    if node.op_type == 'QuantizeLinear':
        print(node.name)
        quantizeLinear_nodes.append(node)
    #    for item in node.input:
    #        print(item)
    elif node.op_type == 'DequantizeLinear':
        print(node.name)
        dequantizeLinear_nodes.append(node)
        

aimet_encoder = EncodingHelper(nodes)

for qu_node in quantizeLinear_nodes:

    de_found = False
    weights_node = False
    for de_node in dequantizeLinear_nodes: 
        if qu_node.output[0] in de_node.input:
            de_found = True
        
        if qu_node.input[0] in initializer_map:
            weights_node = True
            
        consume_node = False
        if de_found:
            for node in nodes:
                for index,name in enumerate(node.input):
                    if de_node.output[0] == name and purne_weights_qat_node and weights_node:
                        print('move {} {}'.format(qu_node.output[0],node.input[0]))
                        
                        node.input[index] = qu_node.input[0]
                        
                        aimet_encoder.onnx_quant_encoding_helper(qu_node,initializer_map,is_act_encoding=False)
                        
                        graph_def.node.remove(de_node)
                        graph_def.node.remove(qu_node)
                        
                        consume_node = True
                        break
                    #    #pass
                    if  de_node.output[0] == name and not weights_node:
                        print('move {} {}'.format(qu_node.output[0],node.input[0]))
                        
                        node.input[index] = qu_node.input[0]
                        
                        aimet_encoder.onnx_quant_encoding_helper(qu_node,initializer_map)
                        
                        graph_def.node.remove(de_node)
                        graph_def.node.remove(qu_node)
                        
                        consume_node = True
                        break
                    
        if consume_node or de_found:
            break

model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)

#do simplify
#model, check = simplify(model)
#assert check, "Simplified ONNX model could not be validated"

onnx.save(model, "convnets_modified.onnx")
aimet_encoder.write_to_aimet_format()
