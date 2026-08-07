
# 处理导入和其他设置。
import torch
from torchvision.models import mobilenet_v2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("use device:",device)

# 实例化一个mobilenet_v2
model = mobilenet_v2(weights='DEFAULT').eval().to(device)
#  
dummy_input = torch.randn((10, 3, 224, 224), device=device)


# 创建一个 QuantizationSimModel，并确保模型包含量化运算。
from aimet_common.defs import QuantScheme
from aimet_common.quantsim_config.utils import get_path_for_per_channel_config
from aimet_torch.quantsim import QuantizationSimModel

# 进行量化模拟的Model,并且让他包含量化操作
sim = QuantizationSimModel(model, 
                           dummy_input,
                           quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                           config_file=get_path_for_per_channel_config(),
                           default_param_bw=8,
                           default_output_bw=16)
# 该模型应由 Quantizednn.Modules 组成，类似于如下所示的输出：
print(sim)

# 第三步：校准模型。此示例使用随机值作为输入。在实际情况下，应使用具有代表性的数据集进行校准。
def forward_pass(model):
    with torch.no_grad():
        model(torch.randn((10, 3, 224, 224), device=device))
#  进行计算编码
sim.compute_encodings(forward_pass)


# 第四步：评估模型。
output = sim.model(dummy_input)
print(output)

