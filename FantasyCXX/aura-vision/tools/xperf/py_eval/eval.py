import argparse
import json
import os
import shutil
import time
import ast

from face_dms_eval import DmsEval
from face_attribute_ir_eval import FaceAttributeIR
from face_attribute_rgb_eval import FaceAttributeRGB
from face_call_eval import FaceCall
from face_cover_eval import FaceCoverEval
from face_emotion_eval import FaceEmotionEval
from face_eye_center_eval import FaceEyeCenter
from face_landmark_eyeclose_eval import FaceLandmarkEyeCloseEval
from face_liveness_ir_eval import FaceLivenessIR
from face_liveness_rgb_eval import FaceLivenessRGB
from face_recognize_eval import FaceRecognizeEval
from gesture_type_eval import GestureTypeEval


def main(bin_path, batch_size, eval_feature, use_local_detect_result):
    data_path = os.path.join(bin_path, 'eval_data')
    os.system('adb shell "mkdir -p ' + data_path + '"')

    start_time= time.time()
    for feature in eval_feature:
        # batch detect buffer path
        buffer_path = feature + '_detect_buffer_image'

        feature_data_path = os.path.join(data_path, feature)
        os.system('adb shell "mkdir ' + feature_data_path + '"')

        # detect result json file
        detect_result_file = feature + '_detect_result.json'

        feature_eval = None
        if feature == 'dms':
            feature_eval = DmsEval()
        elif feature == 'face_call':
            feature_eval = FaceCall()
        elif feature == 'face_liveness_ir':
            feature_eval = FaceLivenessIR()
        elif feature == 'face_liveness_rgb':
            feature_eval = FaceLivenessRGB()
        elif feature == 'face_recognize':
            feature_eval = FaceRecognizeEval(do_error_predict_rate=1)
        elif feature == 'gesture_type':
            feature_eval = GestureTypeEval()
        elif feature == 'face_landmark_eye_close':
            feature_eval = FaceLandmarkEyeCloseEval()
            feature_eval.eval(bin_path, use_local_detect_result)
            return
        elif feature == 'face_attribute_rgb':
            feature_eval = FaceAttributeRGB()
        elif feature == 'face_attribute_ir':
            feature_eval = FaceAttributeIR()
        elif feature == 'face_eye_center':
            feature_eval = FaceEyeCenter()
        elif feature == 'face_cover':
            feature_eval = FaceCoverEval()
        elif feature == 'face_emotion':
            feature_eval = FaceEmotionEval()
        else:
            return

        # split eval images by batch_size
        eval_images = os.listdir(os.path.join(feature_eval.local_data_path, feature_eval.save_file_name))
        eval_images = [image for image in eval_images if not image.startswith('.DS')]
        num_images = len(eval_images)
        batch_list = [batch_size for _ in range(num_images // batch_size)]
        remainder = num_images % batch_size
        if remainder != 0:
            batch_list.append(remainder)
        print('eval image batch list:', batch_list)
        if not use_local_detect_result:
            # detect by batch
            local_batch_buffer_data_path = os.path.join(feature_eval.local_data_path, buffer_path)
            for i, batch in enumerate(batch_list):
                print('rm device buffer image.')
                os.system('adb shell "rm -rf ' + feature_data_path + '/' + buffer_path + '"')

                # batch data
                batch_images = eval_images[i*batch_size:i*batch_size + batch]
                print("batch %d, push %d eval images to device." % (i + 1, batch))

                # push batch eval data to device
                os.system('rm -rf ' + local_batch_buffer_data_path)
                os.mkdir(local_batch_buffer_data_path)
                for image in batch_images:
                    src_image = os.path.join(feature_eval.local_data_path, feature_eval.save_file_name, image)
                    shutil.copyfile(src_image, os.path.join(local_batch_buffer_data_path, image))
                os.system('adb push ' + local_batch_buffer_data_path + ' ' + feature_data_path)
                os.system('rm -rf ' + local_batch_buffer_data_path)

                detect_result_path = os.path.join(feature_data_path, detect_result_file + '-batch' + str(i))
                os.system('adb shell "rm ' + detect_result_path + '"')
                os.system('adb shell "touch ' + detect_result_path + '"')
                os.system('adb shell "echo {} > ' + detect_result_path + '"')

                # do batch detect on device
                print("do detect on device.")
                os.system('adb shell "cd ' + bin_path +
                          ' && export LD_LIBRARY_PATH=' + bin_path + ':${LD_LIBRARY_PATH}' +
                          ' && ./xperf ' + 'accuracy' + ' ' + feature + ' ' + feature_data_path + '/' + buffer_path + ' ' + detect_result_path + '"')
                os.system('adb shell "rm -rf ' + feature_data_path + '/' + buffer_path + '"')

                # get detect result from device
                os.system('adb pull ' + detect_result_path + ' ' + feature_eval.detect_result_path)

        print("feature detect cost time : ", time.time()- start_time)

        predict_json = {}
        for i in range(len(batch_list)):
            detect_result_path = os.path.join(feature_eval.detect_result_path, detect_result_file + '-batch' + str(i))
            batch_json = json.load(open(detect_result_path))
            predict_json.update(batch_json)

        start_time = time.time()
        feature_eval.eval(predict_json)
        feature_eval.print_result()
        feature_eval.write_wrong_record()

        print("feature eval cost time : ", time.time()- start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, help='xperf executable program path')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for eval (default: 1024)')
    parser.add_argument('--eval_feature', nargs='+', help='eval native feature list, support["dms"]')
    parser.add_argument('--use_local_detect_result', type=ast.literal_eval, help='use local json file do eval (default: False)')

    args = parser.parse_args()
    bin_path = args.bin_path
    batch_size = args.batch_size
    eval_feature = args.eval_feature
    use_local_detect_result = args.use_local_detect_result

    start = time.time()
    main(bin_path, batch_size, eval_feature, use_local_detect_result)
    end = time.time()
    print("total feature eval time cost : ", end - start)
