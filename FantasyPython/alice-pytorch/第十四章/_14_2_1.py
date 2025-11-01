# from torchaudio import datasets
#
# datasets.SPEECHCOMMANDS(
#     root="../dataset/",  # 你保存数据的路径
#     url='speech_commands_v0.02',  # 下载数据版本URL
#     folder_in_archive='SpeechCommands',
#     download=True)  # 这个记得选True


import librosa
import cv2
# 使用librosa获得音频的梅尔频谱
wav, sr = librosa.load("../dataset/SpeechCommands/speech_commands_v0.02/bird/0a7c2a8d_nohash_1.wav", sr=32000)		#sr为取样频率
# 计算音频信号的MFCC
spec_image = librosa.feature.mfcc(y=wav, sr=sr)

print(wav.shape)
print(spec_image.shape)
