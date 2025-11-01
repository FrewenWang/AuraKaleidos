import copy
import math
# import paddle
import cv2 as cv
import numpy as np
import os
import onnxruntime


def load_pretrain_weight(model, weight):
    """
    加载预训练模型
    :param model:
    :param weight:
    :return:
    """
    weights_path = weight + '.pdparams'

    if not (os.path.exists(weights_path)):
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(weights_path))

    model_dict = model.state_dict()

    param_state_dict = paddle.load(weights_path)
    ignore_weights = set()

    for name, weight in param_state_dict.items():
        if name in model_dict.keys():
            if list(weight.shape) != list(model_dict[name].shape):
                ignore_weights.add(name)
        else:
            ignore_weights.add(name)

    for weight in ignore_weights:
        param_state_dict.pop(weight, None)

    model.set_dict(param_state_dict)


def main():
    norm_w, norm_h = 72, 72
    model = model_net_caffeMain.Model_caffeMain()

    pathModel = "./FaceLandmark/facept_epoch39.pdparams"
    model_layer_dict = paddle.load(pathModel)
    model.set_state_dict(model_layer_dict)

    allData = []
    pathFile_item = "/media/baidu/3.6TB_SSD/facepts/Faciallandmark/filelist/eval_list.txt"
    infos_tmp = open(pathFile_item).readlines()
    for linet in infos_tmp:
        allData.append(linet.strip())

    image_dir = "/media/baidu/3.6TB_SSD/facepts/Faciallandmark"

    for id in range(len(allData)):
        pathImg = os.path.join(image_dir, allData[id])

        pathImg_info = pathImg.split("/")
        pathImg_info_prefix = "/".join(pathImg_info[:-1])
        filename_img = pathImg_info[-1]
        filename_img_info = filename_img.split(".")
        filename_img_prefix = ".".join(filename_img_info[:-1])
        filename_json = "out_" + filename_img_prefix + ".json"
        pathJson = os.path.join(pathImg_info_prefix, filename_json)
        label = facept_utils.read_106_labels_from_json(pathJson)
        # facepts = np.array(label[:212])
        # heapose = label[212:215] #pitch,yaw,roll
        # eyestatus = label[215]
        label = np.array(label)  # 216维度
        data = cv.imread(pathImg)

        data_out, _ = dataset_augment.random_cropFace(data, label, phase_train=False)
        data_draw = copy.deepcopy(data_out)
        data_out = cv.resize(data_out, (norm_w, norm_h))

        data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)
        data1 = np.array(data, dtype=np.float64)
        m,s = cv.meanStdDev(data)
        data2 = (data1-m)/(1e-6 + s)

        inputdata = np.zeros(shape=[1, 1, norm_h, norm_w])
        inputdata[0][0] = data2
        inputdata2 = paddle.to_tensor(inputdata, dtype=paddle.float32)
        output_rslt = model(inputdata2)
        output_rslt1 = output_rslt.numpy()[0]
        output_rslt1 = list(output_rslt1)

        pitch = output_rslt1[-4] * 180 / math.pi
        yaw = output_rslt1[-3] * 180 / math.pi
        roll = output_rslt1[-2] * 180 / math.pi
        print (output_rslt1[0], "pitch-yaw-roll:", round(pitch,2), round(yaw,2), round(roll,2), output_rslt1[-1])
        data_out_h, data_out_w, _ = data_draw.shape
        for i in range(106):
            px = output_rslt1[2 * i + 1]
            py = output_rslt1[2 * i + 2]
            px = int(px * data_out_w)
            py = int(py * data_out_h)
            cv.circle(data_draw, (px,py), 3, (0,0,255), -1, 8, 0)
        cv.imshow("data_out", data_draw)
        #cv.imwrite("/media/baidu/ssd2/traindemo/facept_demo/" + str(id) + ".jpg", data_draw)
        cv.waitKey(0)


def main_image_onnx(ort_sess, data, rect):
    norm_w, norm_h = 128, 128

    data_out = data[rect[1]:rect[3], rect[0]:rect[2], :]
    data_draw = copy.deepcopy(data_out)
    data_out = cv.resize(data_out, (norm_w, norm_h))

    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)
    m,s = cv.meanStdDev(data)
    data2 = (data-m)/(1e-6 + s)

    inputdata = np.zeros(shape=[1, 1, norm_h, norm_w], dtype=np.float32)
    inputdata[0][0] = data2

    ort_inputs = {ort_sess.get_inputs()[0].name: inputdata}
    output_rslt = ort_sess.run(None, ort_inputs)
    output_rslt1 = list(output_rslt[0][0])

    face_cls = output_rslt1[0]
    pitch = output_rslt1[-4] * 180 / math.pi
    yaw = output_rslt1[-3] * 180 / math.pi
    roll = output_rslt1[-2] * 180 / math.pi
    print(output_rslt1[0], "pitch-yaw-roll:", round(pitch,2), round(yaw,2), round(roll,2), output_rslt1[-1])
    data_out_h, data_out_w, _ = data_draw.shape
    points = []
    for i in range(106):
        px = output_rslt1[2 * i + 1]
        py = output_rslt1[2 * i + 2]
        px = int(px * data_out_w) + rect[0]
        py = int(py * data_out_h) + rect[1]
        #cv.circle(data_draw, (px, py), 3, (0, 0, 255), -1, 8, 0)
        points.append((px, py))
    return points, {"pitch": pitch, "yaw": yaw, "roll": roll}


def main_image(data, rect):
    norm_w, norm_h = 72, 72
    model = model_net_caffeMain.Model_caffeMain()

    pathModel = "./FaceLandmark/facept_epoch39.pdparams"
    model_layer_dict = paddle.load(pathModel)
    model.set_state_dict(model_layer_dict)

    data_out = data[rect[1]:rect[3], rect[0]:rect[2], :]
    data_draw = copy.deepcopy(data_out)
    data_out = cv.resize(data_out, (norm_w, norm_h))

    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)
    data1 = np.array(data, dtype=np.float64)
    m,s = cv.meanStdDev(data)
    data2 = (data1-m)/(1e-6 + s)

    inputdata = np.zeros(shape=[1, 1, norm_h, norm_w])
    inputdata[0][0] = data2
    inputdata2 = paddle.to_tensor(inputdata, dtype=paddle.float32)
    output_rslt = model(inputdata2)
    output_rslt1 = output_rslt.numpy()[0]
    output_rslt1 = list(output_rslt1)

    pitch = output_rslt1[-4] * 180 / math.pi
    yaw = output_rslt1[-3] * 180 / math.pi
    roll = output_rslt1[-2] * 180 / math.pi
    #print(output_rslt1[0], "pitch-yaw-roll:", round(pitch,2), round(yaw,2), round(roll,2), output_rslt1[-1])
    data_out_h, data_out_w, _ = data_draw.shape
    points = []
    for i in range(106):
        px = output_rslt1[2 * i + 1]
        py = output_rslt1[2 * i + 2]
        px = int(px * data_out_w) + rect[0]
        py = int(py * data_out_h) + rect[1]
        #cv.circle(data_draw, (px, py), 3, (0, 0, 255), -1, 8, 0)
        points.append((px, py))
    return points, {"pitch": pitch, "yaw": yaw, "roll": roll}


if __name__ == "__main__":
    main()
