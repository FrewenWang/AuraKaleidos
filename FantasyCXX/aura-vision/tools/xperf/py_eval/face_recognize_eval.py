import os, sys, time, random
import csv
import numpy as np
import shutil

from base_eval import BaseEval


class FaceRecognizeEval(BaseEval):
    def __init__(self, do_error_predict_rate=0.01):
        super(FaceRecognizeEval, self).__init__()
        self.feature_name = 'face_recognize'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'face_recognize_sample_images'
        self.cosine_thresh = 0.51
        self.make_local_path()
        self.prepare_eval_data()
        self.recall_arr = []
        self.error_predict = []
        # 全量计算误检率的耗时很长，do_error_predict_rate:使用多少比例的数据来计算误检率
        self.do_error_predict_rate = do_error_predict_rate

    def prepare_eval_data(self):
        remote_data_path = os.path.join(self.remote_data_path, self.feature_name)
        """
        make sure remote test data is prepared:
        ssh iov@172.20.72.11
        cd /home/iov/vision_space/data/VisionImages/test_images/face_recognize
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

        # load all test images
        with open(os.path.join(self.local_data_path, self.save_file_name + '.csv'), 'r', encoding='utf-8') as f:
            self.test_images = [row[0] for row in list(csv.reader(f))]
            labels = set()
            for test_image in self.test_images:
                label = test_image.split('|')[0]
                labels.add(label)
            self.labels = list(labels)
            self.result_dicts = {label:dict() for label in self.labels}
            print('共有{%d}个ID需要识别' % len(self.labels))

    def eval(self, predict_json):
        os.system('rm -r ' + os.path.join(self.local_data_path, 'face_feature_no_face_images'))
        os.system('mkdir ' + os.path.join(self.local_data_path, 'face_feature_no_face_images'))
        self.no_face = 0
        for test_image in self.test_images:
            if not test_image in predict_json:
                continue
            image_result = predict_json[test_image]
            face_id = int(image_result["face_id"])
            if face_id < 1:
                self.no_face += 1
                shutil.copyfile(os.path.join(self.local_data_path, self.save_file_name, test_image),
                                os.path.join(self.local_data_path, 'face_feature_no_face_images', test_image))
                continue
            face_features = image_result['face_features']
            for label in self.labels:
                if test_image.split('|')[0] != label:
                    continue
                self.result_dicts[label][test_image] = face_features
        print("\r检测图片[%d]张, 无人脸图片[%d]张, 人脸检出率[%.5f]" % (len(self.test_images), self.no_face, 1 - (self.no_face / len(self.test_images))))

        # 计算召回：每个ID的每个样本与该ID的其他样本匹配
        match_predict_total_count = 0
        for label, label_results_dict in self.result_dicts.items():
            label_results = list(label_results_dict.values())
            for i in range(len(label_results) - 1):
                for j in range(i+1, len(label_results)):
                    match_predict_total_count += 1
        print("匹配率计算共需比对次数：%d" % match_predict_total_count)
        match_predict_processed_count = 0
        for label, label_results_dict in self.result_dicts.items():
            label_results = list(label_results_dict.values())
            for i in range(len(label_results) - 1):
                for j in range(i+1, len(label_results)):
                    match_predict_processed_count += 1
                    sys.stdout.write('\r匹配率计算完成：%.2f%%' % (match_predict_processed_count / match_predict_total_count * 100))
                    sys.stdout.flush()
                    cosine_similarity = self.compare_faces(np.asarray(label_results[i], dtype=float),
                                                           np.asarray(label_results[j], dtype=float))
                    if cosine_similarity >= self.cosine_thresh:
                        self.recall_arr.append(1)
                    else:
                        self.recall_arr.append(0)
        print('通过次数：%d，比对次数：%d' % (np.sum(self.recall_arr), len(self.recall_arr)))
        print('匹配通过率：%.5f' % (np.sum(self.recall_arr) / len(self.recall_arr)))

        # 计算误检：每个ID的每个样本与其他ID分别匹配
        total_ids = list(self.result_dicts.keys())
        error_predict_total_count = 0
        for i in range(len(total_ids) - 1):
            for j in range(i+1, len(total_ids)):
                compare_id_1_dict = self.result_dicts[total_ids[i]]
                compare_id_2_dict = self.result_dicts[total_ids[j]]
                for _ in compare_id_1_dict.values():
                    for _ in compare_id_2_dict.values():
                        error_predict_total_count += 1
        do_error_predict_count = error_predict_total_count * self.do_error_predict_rate
        print("\r误检率计算共需比对次数：%d" % error_predict_total_count)
        print("计算比例：%.4f, 共需比对次数：%d" % (self.do_error_predict_rate, do_error_predict_count))
        error_predict_processed_count = 0
        for i in range(len(total_ids) - 1):
            for j in range(i+1, len(total_ids)):
                compare_id_1_dict = self.result_dicts[total_ids[i]]
                compare_id_2_dict = self.result_dicts[total_ids[j]]
                for compare_id_1_result in compare_id_1_dict.values():
                    for compare_id_2_result in compare_id_2_dict.values():
                        if random.randint(1,10000) > (10000 * self.do_error_predict_rate):
                            continue
                        error_predict_processed_count += 1
                        sys.stdout.write('\r误检率计算%d完成：%.2f%%' % (error_predict_processed_count, error_predict_processed_count / do_error_predict_count * 100))
                        sys.stdout.flush()
                        cosine_similarity = self.compare_faces(np.asarray(compare_id_1_result, dtype=float),
                                                               np.asarray(compare_id_2_result, dtype=float))
                        if cosine_similarity >= self.cosine_thresh:
                            self.error_predict.append(1)
                        else:
                            self.error_predict.append(0)

        print('误检次数：%d，比对次数：%d' % (np.sum(self.error_predict), len(self.error_predict)))
        print('误检率：%.7f' % (np.sum(self.error_predict) / len(self.error_predict)))

    @staticmethod
    def compare_faces(face1st, face2nd):
        num = face1st.dot(face2nd.T)
        denom = np.linalg.norm(face1st) * np.linalg.norm(face2nd)
        return num / denom

    def print_result(self):
        pass

    def write_wrong_record(self):
        pass


