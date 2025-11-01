# -*- coding:utf8 -*-
import cv2
import os

def get_frame_from_video(video_name, prefix_path, sample_rate):
    """
    Args:
        video_name:输入视频名字
        prefix_path: 保存图片的帧率间隔
        sample_rate: 抽帧的频率
    Returns:
    """
    print(f'video_name: {video_name} ')
    # 开始读视频
    video_capture = cv2.VideoCapture(video_name)
    # 读取视频的 FPS ,便于确定抽帧频次
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    if sample_rate >= fps or sample_rate <= 0:
        sample_rate = fps
    print(f'video extract frames fps: {fps}, sample_rate: {sample_rate}')

    d_interval = float(fps)/float(sample_rate)
    f_index = 0
    s_index = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            print('video is all read')
            break
        if int(f_index % d_interval) == 0:
            save_name = os.path.join(prefix_path, 'select-'+str(s_index)+'_from-'+str(f_index) +
                                                  '_sample_rate-' + str(sample_rate) + '_fps-' + str(fps) + '.jpg')
            print(f'save_name: {save_name}')
            cv2.imwrite(save_name, frame)
            s_index += 1
        f_index += 1        
# 获取文件夹下所有文件名
def get_video_files(path: str, file_list: list):
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isdir(file_path):
                continue
            if not file_util.is_end_with(file_path, 'mp4'):
                continue
            file_list.append(path + file_name)
    else:
        file_list.append(path)
    file_list.sort()


if __name__ == '__main__':
    # 默认值为0， 即关闭抽帧频率。
    # 如果设置为1，则为每秒抽取1帧，以此类推，设置为30，则每秒抽取30帧。
    # 如果超过视频帧率，则按照视频帧率进行抽取
    frame_sample_rate = 0
    data_path = "/home/baiduiov/02.ProjectSpace/03.JIDU/售后问题/20240805/"
    video_files = list()
    get_video_files(data_path, video_files)
    total = len(video_files)
    print(f'total videos: {total}')
    for i in range(len(video_files)):
        try:
            video_name = video_files[i]
            prefix_path = video_name.split('.mp4')[0]
            if not os.path.exists(video_name.split('.mp4')[0]):
                os.mkdir(video_name.split('.mp4')[0])
            print(f'video_name: {video_name}, prefix_path:{prefix_path}')
            get_frame_from_video(video_name, prefix_path, frame_sample_rate)
        except Exception as e:
            print(f'video extract frames error: {e}')
