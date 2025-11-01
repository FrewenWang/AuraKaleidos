import io
import json
# 安装所需工具包
import flask
import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50
from torch.autograd import Variable

# 初始化Flask app
app = flask.Flask(__name__)
model = None
# 如果你使用的是Mac系统、或者你的电脑没有nvidia的显卡的GPU。
# 那么你需要关闭这个开关
use_gpu = False

# 返回结果用的
# 这个文件里面存储的事resnet的分类标签
with open('imagenet_class.txt', 'r') as f:
    idx2label = eval(f.read())


# 加载模型进来
def load_model():
    """Load the pre-trained model, you can use your model just as easily.

    """
    global model
    # resnet50 直接下载的模型，我们可以把模型的下载
    # 我们将模型的权重文件下载
    model = resnet50(pretrained=True)
    # 将模型制定到测试的情况
    model.eval()
    if use_gpu:
        model.cuda()


# 数据预处理
def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.
    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    # 预处理一： 确认图像是否是BGR，如果不是将图像转成BGR
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    # 进行图像的resize
    image = T.Resize(target_size)(image)
    # 将图片数据转换成为tensor
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize. mean与std
    # 进行图像的归一化操作。
    # TODO 这个地方均值为什么是0.485, 0.456, 0.406。标准差为什么是0.229, 0.224, 0.225
    # 这个就是模型训练集的数据的均值和方差
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    #
    return Variable(image, volatile=True)  # 不需要求导


# 开启服务
@app.route("/predict", methods=["POST"])
def predict():
    """

    Returns:
    """
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):

            # Read the image in PIL format
            # 把image字段数据读取进来
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))  # 二进制数据

            # Preprocess the image and prepare it for classification.
            #  进行预处理操作
            image = prepare_image(image, target_size=(224, 224))

            # Classify the input image and then initialize the list of predictions to return to the client.
            # 针对推理结果
            preds = F.softmax(model(image), dim=1)
            # 判断哪三个类别概率是最大的
            results = torch.topk(preds.cpu().data, k=3, dim=1)
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())

            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                label_name = idx2label[label]
                r = {"label": label_name, "probability": float(prob)}
                data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()
