import os
import csv
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from base_eval import BaseEval


class FaceAttributeIR(BaseEval):
    def __init__(self):
        super(FaceAttributeIR, self).__init__()
        self.feature_name = 'face_attribute_ir'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'face_attribute_ir_sample_images'

        self.class_names_glasses = ["color", "no", "nocolor"]
        self.actual_glasses      = []
        self.predicted_glasses   = []
        self.test_dict_glasses   = {}

        self.class_names_race    = ["Black", "White", "Yellow"]
        self.actual_race         = []
        self.predicted_race      = []
        self.test_dict_race      = {}

        self.class_names_age     = ["teenager", "young", "midlife", "senior"]
        self.actual_age          = []
        self.predicted_age       = []
        self.test_dict_age       = {}

        self.class_names_gender  = ["male", "female"]
        self.actual_gender       = []
        self.predicted_gender    = []
        self.test_dict_gender    = {}

        self.make_local_path()
        self.prepare_eval_data()

    def prepare_eval_data(self):
        remote_data_path = os.path.join(self.remote_data_path, 'face_attribute')
        """
        make sure remote test data is prepared:
        ssh iov@172.20.72.11
        cd /home/iov/vision_space/data/VisionImages/test_images/face_attribute_ir
        python random_select.py --sample_rate=<sample_rate> --random_seed=[random_seed]
        python data_process.py
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
            self.test_dict_gender  = {row[0]: int(row[1]) for row in test_data}
            self.test_dict_age     = {row[0]: int(row[2]) for row in test_data}
            self.test_dict_race    = {row[0]: int(row[3]) for row in test_data}
            self.test_dict_glasses = {row[0]: int(row[4]) for row in test_data}

    def eval(self, predict_json):
        self.no_face = 0
        for image_name in self.test_dict_gender.keys():
            if not image_name in predict_json:
                continue
            image_result_json = predict_json[image_name]
            face_id = int(image_result_json["face_id"])
            if face_id < 1:
                self.no_face += 1
                continue
            image_label_gender   = self.test_dict_gender[image_name]
            image_label_age      = self.test_dict_age[image_name]
            image_label_race     = self.test_dict_race[image_name]
            image_label_glasses  = self.test_dict_glasses[image_name]

            detect_label_gender  = int(image_result_json["gender_state"])
            detect_label_glasses = int(image_result_json["glass_state"])
            detect_label_race    = int(image_result_json["race_state"])
            detect_label_age     = int(image_result_json["age_state"])

            self.actual_gender.append(image_label_gender)
            self.predicted_gender.append(detect_label_gender)

            self.actual_age.append(image_label_age)
            self.predicted_age.append(detect_label_age)

            self.actual_glasses.append(image_label_glasses)
            self.predicted_glasses.append(detect_label_glasses)

            self.actual_race.append(image_label_race)
            self.predicted_race.append(detect_label_race)

    def print_result(self):
        print("检测图片[%d]张, 无人脸图片[%d]张, 人脸检出率[%.5f]" % (len(self.test_dict_gender), self.no_face, 1 - (self.no_face / len(self.test_dict_gender))))

        print_format = "{0:^10}\t{1:^10}\t{2:^10}"

        print('--------------------gender-------------------')
        confuse_matrix_gender = confusion_matrix(self.actual_gender, self.predicted_gender)
        print(confuse_matrix_gender)
        # print(print_format.format("类别", "precision", "recall"))
        # for i in range(len(confuse_matrix_gender)):
        #     if i >= len(self.class_names_gender):
        #         break
        #     precision =  confuse_matrix_gender[i][i] / np.sum(confuse_matrix_gender[:, i])
        #     recall    =  confuse_matrix_gender[i][i] / np.sum(confuse_matrix_gender[i])
        #     print(print_format.format(self.class_names_gender[i], '%.4f' % precision, '%.4f' % recall))
        report = classification_report(self.actual_gender, self.predicted_gender, target_names=self.class_names_gender)
        print(report)

        print('--------------------age-------------------')
        confuse_matrix_age = confusion_matrix(self.actual_age, self.predicted_age)
        print(confuse_matrix_age)
        print(print_format.format("类别", "precision", "recall"))
        for i in range(1, len(confuse_matrix_age)):
            if i >= len(self.class_names_age):
                break
            precision =  confuse_matrix_age[i][i] / np.sum(confuse_matrix_age[:, i])
            recall    =  confuse_matrix_age[i][i] / np.sum(confuse_matrix_age[i])
            print(print_format.format(self.class_names_age[i], '%.4f' % precision, '%.4f' % recall))
        # report = classification_report(self.actual_age, self.predicted_age, target_names=self.class_names_age)
        # print(report)

        print('--------------------race-------------------')
        confuse_matrix_race = confusion_matrix(self.actual_race, self.predicted_race)
        print(confuse_matrix_race)
        # print(print_format.format("类别", "precision", "recall"))
        # for i in range(len(confuse_matrix_race)):
        #     if i >= len(self.class_names_race):
        #         break
        #     precision =  confuse_matrix_race[i][i] / np.sum(confuse_matrix_race[:, i])
        #     recall    =  confuse_matrix_race[i][i] / np.sum(confuse_matrix_race[i])
        #     print(print_format.format(self.class_names_race[i], '%.4f' % precision, '%.4f' % recall))
        report = classification_report(self.actual_race, self.predicted_race, target_names=self.class_names_race)
        print(report)

        print('--------------------glasses-------------------')
        confuse_matrix_glasses = confusion_matrix(self.actual_glasses, self.predicted_glasses)
        print(confuse_matrix_glasses)
        print(print_format.format("类别", "precision", "recall"))
        # for i in range(len(confuse_matrix_glasses)):
        #     if i >= len(self.class_names_glasses):
        #         break
        #     precision =  confuse_matrix_glasses[i][i] / np.sum(confuse_matrix_glasses[:, i])
        #     recall    =  confuse_matrix_glasses[i][i] / np.sum(confuse_matrix_glasses[i])
        #     print(print_format.format(self.class_names_glasses[i], '%.4f' % precision, '%.4f' % recall))
        report = classification_report(self.actual_glasses, self.predicted_glasses, target_names=self.class_names_glasses)
        print(report)

    def write_wrong_record(self):
        pass

