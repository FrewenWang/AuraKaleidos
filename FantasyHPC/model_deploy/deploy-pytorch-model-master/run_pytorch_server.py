import io
from ast import literal_eval
from pathlib import Path

# 安装所需工具包
import flask
import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# 初始化Flask app
app = flask.Flask(__name__)
model = None
use_gpu = False

# 返回结果用的
CLASS_NAMES_PATH = Path(__file__).with_name("imagenet_class.txt")
with CLASS_NAMES_PATH.open(encoding="utf-8") as class_names_file:
    idx2label = literal_eval(class_names_file.read())

# 加载模型进来


def load_model():
    """Load the pre-trained model, you can use your model just as easily."""
    global model
    model = resnet50(pretrained=True)
    model.eval()
    if use_gpu:
        model.cuda()


# 数据预处理


def prepare_image(image, target_size):
    """
    Do image preprocessing before prediction on any data.
    图像预处理
    :param image:       original image
    :param target_size: target image size
    :return:
            preprocessed image
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)

    # Convert to Torch.Tensor and normalize. mean与std
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
        image
    )

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return image


# 开启服务
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == "POST" and flask.request.files.get("image"):
        # Read the image in PIL format
        image = flask.request.files["image"].read()
        image = Image.open(io.BytesIO(image))  # 二进制数据

        # Preprocess the image and prepare it for classification.
        # 预处理函数进行预处理
        image = prepare_image(image, target_size=(224, 224))

        # Classify the input image and then initialize the list of predictions to return to the client.
        with torch.no_grad():
            preds = functional.softmax(model(image), dim=1)
        # 获取最高的三个概率
        results = torch.topk(preds.cpu().data, k=3, dim=1)
        results = (results[0].cpu().numpy(), results[1].cpu().numpy())

        data["predictions"] = list()

        # Loop over the results and add them to the list of returned predictions
        for prob, label in zip(results[0][0], results[1][0], strict=True):
            label_name = idx2label[label]
            r = {"label": label_name, "probability": float(prob)}
            data["predictions"].append(r)

        # Indicate that the request was a success.
        data["success"] = True

    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)


if __name__ == "__main__":
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    app.run()
