import os
import csv
import copy
import shutil
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from base_eval import BaseEval


class GestureTypeEval(BaseEval):
    def __init__(self):
        super(GestureTypeEval, self).__init__()
        self.feature_name = 'gesture_type'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'gesture_type_sample_images'
        self.class_names = ['quan',
                            '1', '2', '3', '4', '5',
                            'ok', 'zan', 'shang', 'xia']
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
        cd /home/iov/vision_space/data/VisionImages/test_images/gesture_type
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
        f1 = open(os.path.join(self.local_data_path, 'gesture_type_detect_wrong_images.csv'), 'w', encoding='utf-8');
        csv_writer = csv.writer(f1)
        self.no_gesture = 0
        for image_name, image_label in self.test_dict.items():
            image_result_json = predict_json[image_name]
            detect_label = int(image_result_json["gesture_type"])
            if detect_label < 0 or detect_label > 10:
                self.no_gesture += 1
                continue
            if detect_label == 10 or detect_label == 9:
                detect_label -= 1

            self.actual.append(image_label)
            self.predicted.append(detect_label)
            if image_label != detect_label:
                csv_writer.writerow([image_name, image_label])
                self.wrong_detect.append((image_name,
                                          self.class_names[image_label],
                                          self.class_names[detect_label]))
        f1.close()
        print("\r检测图片[%d]张, 无手势图片[%d]张, 手势检出率[%.5f]" % (len(self.test_dict.items()), self.no_gesture, 1 - (self.no_gesture / len(self.test_dict.items()))))

    def print_result(self):
        confuse_matrix = confusion_matrix(self.actual, self.predicted)
        print_format = "{0:^10}\t{1:^10}\t{2:^10}"
        print(print_format.format("类别", "precision", "recall"))
        for i in range(len(confuse_matrix)):
            precision =  confuse_matrix[i][i] / np.sum(confuse_matrix[:, i])
            recall    =  confuse_matrix[i][i] / np.sum(confuse_matrix[i])
            print(print_format.format(self.class_names[i], '%.4f' % precision, '%.4f' % recall))

        report = classification_report(self.actual, self.predicted, target_names=self.class_names)
        print(report)

    def write_wrong_record(self):
        with open(os.path.join(self.local_data_path, 'gesture_type_wrong_detect.csv'), 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["image", "image label", "predict label"])
            os.system('rm -r ' + os.path.join(self.local_data_path, 'gesture_type_detect_wrong_images'))
            os.system('mkdir ' + os.path.join(self.local_data_path, 'gesture_type_detect_wrong_images'))
            for record in self.wrong_detect:
                csv_writer.writerow([record[0], record[1], record[2]])
                shutil.copyfile(os.path.join(self.local_data_path, self.save_file_name, record[0]),
                                os.path.join(self.local_data_path, 'gesture_type_detect_wrong_images', record[0]))

