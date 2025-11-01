import os
import csv
import numpy as np
import shutil

from base_eval import BaseEval


class FaceLandmarkEyeCloseEval(BaseEval):
    def __init__(self):
        super(FaceLandmarkEyeCloseEval, self).__init__()
        self.feature_name = 'face_landmark_eye_close'
        self.local_data_path = self.local_data_path + '/' + self.feature_name
        self.save_file_name = 'close_eyes_images'
        self.cosine_thresh = 0.51
        self.make_local_path()
        self.prepare_eval_data()

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

    def eval(self, bin_path, no_remote_detect):
        # push测试集到设备
        data_path = os.path.join(bin_path, 'eval_data', self.feature_name)
        detect_result_path = os.path.join(data_path, 'detect_result')
        # 执行检测
        if not no_remote_detect:
            os.system('adb shell "mkdir -p ' + data_path + '"')
            print('adb push ' + os.path.join(self.local_data_path, self.save_file_name) + ' ' + data_path)
            os.system('adb push ' + os.path.join(self.local_data_path, self.save_file_name) + ' ' + data_path)
            os.system('adb shell "rm -r ' + detect_result_path + '"')
            os.system('adb shell "mkdir ' + detect_result_path + '"')
            print("do detect on device.")
            print('export LD_LIBRARY_PATH=' + bin_path + ':${LD_LIBRARY_PATH}')
            print('./xperf ' + 'accuracy' + ' ' + self.feature_name + ' ' + data_path + '/' + self.save_file_name + ' ' + detect_result_path + '"')
            os.system('adb shell "cd ' + bin_path +
                      ' && export LD_LIBRARY_PATH=' + bin_path + ':${LD_LIBRARY_PATH}' +
                      ' && ./xperf ' + 'accuracy' + ' ' + self.feature_name + ' ' + data_path + '/' + self.save_file_name + ' ' + detect_result_path + '"')
            os.system('adb shell "rm -rf ' + data_path + '/' + self.save_file_name + '"')
        # 拷贝结果图片到本地
        os.system('adb pull ' + detect_result_path + ' ' + self.local_data_path)
        pass

    def print_result(self):
        pass

    def write_wrong_record(self):
        pass


