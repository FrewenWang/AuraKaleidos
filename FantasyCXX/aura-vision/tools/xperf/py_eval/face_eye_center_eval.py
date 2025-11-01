import os
import csv
import numpy as np

from base_eval import BaseEval


class FaceEyeCenter(BaseEval):
    def __init__(self):
        super(FaceEyeCenter, self).__init__()
        self.feature_name = 'face_eye_center'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'face_eye_center_sample_images'

        self.eye_centroid_distance = []
        self.eye_lmks_distance = []
        self.test_dict = {}

        self.make_local_path()
        self.prepare_eval_data()

    def prepare_eval_data(self):
        remote_data_path = os.path.join(self.remote_data_path, self.feature_name)
        """
        make sure remote test data is prepared:
        ssh iov@172.20.72.11
        cd /home/iov/vision_space/data/VisionImages/test_images/face_eye_center
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
            self.test_dict = {row[0]: row[1:] for row in test_data}

    def eval(self, predict_json):
        self.no_eye = 0
        for image_name, eye_infos in self.test_dict.items():
            if not image_name in predict_json:
                continue
            image_result_json = predict_json[image_name]
            predict_eye_centroid = [float(image_result_json["eye_centroid_x"]), float(image_result_json["eye_centroid_y"])]
            predict_eye_lmks = image_result_json["eye_lmks"]

            actual_eye_centroid = eye_infos[1:3]
            actual_eye_centroid = [float(info) for info in actual_eye_centroid]
            actual_eye_lmks = eye_infos[3:]
            actual_eye_lmks = [float(info.replace('\n', '')) for info in actual_eye_lmks]

            eye_lmk_distance = 0.
            for i in range(8):
                lmk_distance = self.vectors_euclidean_distance(np.asarray(predict_eye_lmks[i * 2:i * 2 + 2], dtype=float),
                                             np.asarray(actual_eye_lmks[i * 2:i * 2 + 2], dtype=float))
                eye_lmk_distance += lmk_distance
            self.eye_lmks_distance.append(eye_lmk_distance / 8)

            centroid_distance = self.vectors_euclidean_distance(np.asarray(predict_eye_centroid, dtype=float),
                                                             np.asarray(actual_eye_centroid, dtype=float))
            self.eye_centroid_distance.append(centroid_distance)

    def print_result(self):
        # print("检测图片[%d]张, 无人脸图片[%d]张, 人脸检出率[%.5f]" % (len(self.test_dict), self.no_face, 1 - (self.no_face / len(self.test_dict))))
        print("瞳孔平均欧式距离：%.5f" % (np.mean(self.eye_centroid_distance)))
        print("眼皮关键点平均欧式距离：%.5f" % (np.mean(self.eye_lmks_distance)))


    def write_wrong_record(self):
        pass

    @staticmethod
    def vectors_cosine_distance(v1, v2):
        num = v1.dot(v2.T)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return num / (denom + 1e-6)

    @staticmethod
    def vectors_euclidean_distance(v1, v2):
        return np.sqrt(np.sum(np.square(v1 - v2)))