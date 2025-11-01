import os


class BaseEval(object):
    def __init__(self):
        self.remote_addr = 'iov@172.20.72.11'
        self.remote_data_path = '/home/iov/vision_space/data/VisionImages/test_images'
        self.local_data_path = './test_data'

    def make_local_path(self):
        if not os.path.exists(self.local_data_path):
            os.system('mkdir -p ' + self.local_data_path)
        self.detect_result_path = os.path.join(self.local_data_path, 'detect_result')
        if not os.path.exists(self.detect_result_path):
            os.mkdir(self.detect_result_path)

    def prepare_eval_data(self):
        pass

    def eval(self, predict_json):
        pass

    def print_result(self):
        pass

    def write_wrong_record(self):
        pass