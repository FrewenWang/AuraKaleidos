import os
import numpy as np


from base_eval import BaseEval


class FaceCoverEval(BaseEval):
    def __init__(self):
        super(FaceCoverEval, self).__init__()
        self.feature_name = 'face_cover'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'face_cover_sample_images'
        self.cover_mouth_predicted = []
        self.mask_predicted        = []
        self.imgs_predicted        = []
        self.test_list = {}
        self.make_local_path()
        self.prepare_eval_data()

    def prepare_eval_data(self):
        remote_data_path = os.path.join(self.remote_data_path, self.feature_name)
        """
        make sure remote test data is prepared:
        ssh iov@172.20.72.11
        cd /home/iov/vision_space/data/VisionImages/test_images/face_cover
        python random_select.py --sample_rate=<sample_rate> --random_seed=[random_seed]
        """
        # pull sample test images to local
        test_sample_tar = os.path.join(remote_data_path, self.save_file_name + '.tar.gz')
        if not os.path.exists(os.path.join(self.local_data_path, self.save_file_name)):
            print('pull eval image from remote.')
            os.system('scp ' + self.remote_addr + ':' + test_sample_tar + ' ' + self.local_data_path)
            os.system('tar -xvf ' + os.path.join(self.local_data_path, self.save_file_name + '.tar.gz') + ' -C ' + self.local_data_path)

        # build test list
        self.test_list = os.listdir(os.path.join(self.local_data_path, self.save_file_name))

    def eval(self, predict_json):
        self.no_face = 0
        self.cover_mouth_no_face = 0
        self.mask_no_face = 0
        self.imgs_no_face = 0
        self.cover_mouth = 0
        self.mask = 0
        self.imgs = 0
        for image_name in self.test_list:
            if not image_name in predict_json:
                continue
            image_result_json = predict_json[image_name]
            detect_label = int(image_result_json["face_cover_state"])
            face_id = int(image_result_json["face_id"])
            if image_name.startswith('cover_mouth'):
                self.cover_mouth += 1
                if face_id < 1:
                    self.no_face += 1
                    self.cover_mouth_no_face += 1
                else:
                    self.cover_mouth_predicted.append(detect_label)
            elif image_name.startswith('mask'):
                self.mask += 1
                if face_id < 1:
                    self.no_face += 1
                    self.mask_no_face += 1
                else:
                    self.mask_predicted.append(detect_label)
            elif image_name.startswith('imgs'):
                self.imgs += 1
                if face_id < 1:
                    self.no_face += 1
                    self.imgs_no_face += 1
                else:
                    self.imgs_predicted.append(detect_label)

    def print_result(self):
        print("检测图片[%d]张, 无人脸图片[%d]张, 人脸检出率[%.5f]" % (len(self.test_list), self.no_face, 1 - (self.no_face / len(self.test_list))))

        print("捂嘴数据集人脸检出率：%.5f" % (1 - self.cover_mouth_no_face / self.cover_mouth))
        print("捂嘴数据集检出率：%.5f" % (np.sum(self.cover_mouth_predicted) / len(self.cover_mouth_predicted)))
        print("口罩数据集人脸检出率：%.5f" % (1 - self.mask_no_face / self.mask))
        print("口罩数据集检出率：%.5f" % (np.sum(self.mask_predicted) / len(self.mask_predicted)))
        print("无遮挡数据集人脸检出率：%.5f" % (1 - self.imgs_no_face / self.imgs))
        print("无遮挡数据集误检率：%.5f" % (np.sum(self.imgs_predicted) / len(self.imgs_predicted)))

    def write_wrong_record(self):
        pass

