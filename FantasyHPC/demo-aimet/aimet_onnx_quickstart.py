import os

import onnx
import torch
from aimet_onnx import int8, int16
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import make_dummy_input
from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights="DEFAULT").eval()

dummy_input = torch.randn((10, 3, 224, 224))
file_path = os.path.join("/tmp", "mobilenet_v2.onnx")
torch.onnx.export(model, dummy_input, file_path, dynamo=False)
onnx_model = onnx.load_model(file_path)

sim = QuantizationSimModel(onnx_model, param_type=int8, activation_type=int16)


calibration_data = make_dummy_input(onnx_model)
sim.compute_encodings(inputs=[calibration_data])


input_name = tuple(calibration_data.keys())[0]
output = sim.session.run(None, {input_name: dummy_input.numpy()})
print(output)
