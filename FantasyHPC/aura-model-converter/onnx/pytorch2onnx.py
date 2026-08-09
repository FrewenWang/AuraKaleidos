import torch
import torchvision

model = torchvision.models.resnet18(pretrained=False)

device = 'cuda' if torch.cuda.is_available else 'cpu'

dummy_input = torch.randn(1, 3, 224, 224, device=device)
model.to(device)
model.eval()
output = model(dummy_input)

print("pytorch result:", torch.argmax(output))

import torch.onnx

torch.onnx.export(model, dummy_input, './model.onnx', input_names=["input"], output_names=["output"],
                  do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=14,
                  dynamic_axes={"input": {0: "nBatchSize"}, "output": {0: "nBatchSize"}})

# 一般情况
# torch.onnx.export(model, torch.randn(1, c, nHeight, nWidth, device="cuda"), './model.onnx', input_names=["x"], output_names=["y", "z"], do_constant_folding=True, verbose=True, keep_initializers_as_inputs=True, opset_version=14, dynamic_axes={"x": {0: "nBatchSize"}, "z": {0: "nBatchSize"}})

import onnx
import numpy as np
import onnxruntime as ort

model_onnx_path = './model.onnx'
# 验证模型的合法性
onnx_model = onnx.load(model_onnx_path)
onnx.checker.check_model(onnx_model)
# 创建ONNX运行时会话
ort_session = ort.InferenceSession(model_onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# 准备输入数据
input_data = {
    'input': dummy_input.cpu().numpy()
}
# 运行推理
y_pred_onnx = ort_session.run(None, input_data)
print("onnx result:", np.argmax(y_pred_onnx[0]))
