import librosa
import os
import torch
import numpy as np

labels = ["bed","bird","dog","cat","yes"]

image_list = []
label_list = []
for label in labels:
    path = f"../dataset/SpeechCommands/speech_commands_v0.02/{label}"
    for file_name in os.listdir(path):
        file_path = path + "/" + file_name
        wav, sr = librosa.load(file_path, sr=32000)
        spec_image = librosa.feature.mfcc(y=wav, sr=sr)
        spec_image = np.pad(spec_image, ((0, 0), (0, 63 - spec_image.shape[1])), 'constant')
        image_list.append(spec_image)
        label_list.append(labels.index(label))



class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_data,label_list):
        self.input_data = input_data
        self.label_list = label_list
    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data

        input_token = self.input_data[idx]
        output_token = self.label_list[idx]

        input_token = torch.tensor(input_token)
        output_token = torch.tensor(output_token)
        return input_token, output_token


