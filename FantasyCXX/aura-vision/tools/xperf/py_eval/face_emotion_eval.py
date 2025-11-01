import os
import csv
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from base_eval import BaseEval


class FaceEmotionEval(BaseEval):
    def __init__(self):
        super(FaceEmotionEval, self).__init__()
        self.feature_name = 'face_emotion'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'face_emotion_sample_images'
        self.class_names = ["normal", "like", "dislike", "surprise"]
        self.actual = []
        self.predicted = []
        self.test_dict = {}
        self.wrong_detect = []
        self.make_local_path()
        self.prepare_eval_data()

    def prepare_eval_data(self):
        remote_data_path = os.path.join(self.remote_data_path, self.feature_name)
        """
        make sure remote test data is prepared:
        ssh iov@172.20.72.11
        cd /home/iov/vision_space/data/VisionImages/test_images/face_emotion
        python random_select.py --sample_rate=<sample_rate> --random_seed=[random_seed]
        """
        # pull sample test images to local
        test_sample_tar = os.path.join(remote_data_path, self.save_file_name + '.tar.gz')
        if not os.path.exists(os.path.join(self.local_data_path, self.save_file_name)):
            print('pull eval image from remote.')
            os.system('scp ' + self.remote_addr + ':' + test_sample_tar + ' ' + self.local_data_path)
            os.system('tar -xvf ' + os.path.join(self.local_data_path, self.save_file_name + '.tar.gz') + ' -C ' + self.local_data_path)
        # pull sample images label csv to local
        test_sample_csv = os.path.join(remote_data_path, self.save_file_name + '.csv')
        if not os.path.exists(os.path.join(self.local_data_path, self.save_file_name + '.csv')):
            print('pull eval csv from remote.')
            os.system('scp ' + self.remote_addr + ':' + test_sample_csv + ' ' + self.local_data_path)

        # build test dict
        with open(os.path.join(self.local_data_path, self.save_file_name + '.csv'), 'r', encoding='utf-8') as f:
            test_data = list(csv.reader(f))
            self.test_dict = {row[0]: int(row[1]) for row in test_data}

    def eval(self, predict_json):
        self.no_face = 0

        self.dislike_predict = 0
        self.biezui_actual = 0

        self.smile_actual = 0
        self.like_predict = 0
        self.surprise_predict = 0

        for image_name, image_label in self.test_dict.items():
            image_result_json = predict_json[image_name]
            face_id = int(image_result_json["face_id"])
            if face_id < 1:
                self.no_face += 1
                continue
            detect_label = int(image_result_json["face_emotion_state"])
            if detect_label < 0:
                continue
            self.actual.append(image_label)
            self.predicted.append(detect_label)

            # biezuizhoumei
            if image_label == 2:
                self.biezui_actual += 1
                if detect_label == 2:
                    self.dislike_predict += 1

            # zhangzuixiao
            if image_label == 1:
                self.smile_actual += 1
                if detect_label == 1:
                    self.like_predict += 1
                elif detect_label == 3:
                    self.surprise_predict += 1

    def print_result(self):
        print("检测图片[%d]张, 无人脸图片[%d]张, 人脸检出率[%.5f]" % (len(self.test_dict), self.no_face, 1 - (self.no_face / len(self.test_dict))))
        confuse_matrix = confusion_matrix(self.actual, self.predicted)
        print(confuse_matrix)
        print_format = "{0:^10}\t{1:^10}\t{2:^10}"
        print(print_format.format("类别", "precision", "recall"))
        for i in range(len(confuse_matrix)):
            if i >= len(self.class_names):
                break
            precision =  confuse_matrix[i][i] / np.sum(confuse_matrix[:, i])
            recall    =  confuse_matrix[i][i] / np.sum(confuse_matrix[i])
            print(print_format.format(self.class_names[i], '%.4f' % precision, '%.4f' % recall))

        report = classification_report(self.actual, self.predicted, target_names=self.class_names)
        print(report)

        print("撇嘴皱眉识别为dislike:%.5f"    %  (self.dislike_predict / self.biezui_actual))
        print("张嘴笑识别为like:%.5f"         %  (self.like_predict / self.smile_actual))
        print("张嘴笑识别为surprise:%.5f"     %  (self.surprise_predict / self.smile_actual))

    def write_wrong_record(self):
        pass

