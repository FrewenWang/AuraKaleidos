import os
import math
import random
import struct
import numpy as np

def random_reader_creator(input_shape, buffer_size):
    def reader():
        while True:
            images = np.ndarray(input_shape)
            if len(input_shape) == 2:
                rows, cols = input_shape
                images = np.random.random(size=(buffer_size, rows, cols)).astype('float32')
            elif len(input_shape) == 3:
                channels, rows, cols = input_shape
                images = np.random.random(size=(buffer_size, channels, rows, cols)).astype('float32')

            labels = np.random.random_integers(size=(buffer_size, ), low=0, high=9).astype('int64')

            for i in range(buffer_size):
                yield images[i, :], int(labels[i])

    return reader

def calib_reader_creator(input_shape, file_list, data_dir, buffer_size, shuffle=False):
    def reader():
        full_images = []
        try:
            with open(file_list) as flist:
                full_lines = [line.strip() for line in flist]
                if shuffle:
                    np.random.shuffle(full_lines)

                idx = 0
                for line in full_lines:
                    # img_path, label = line.split()
                    img_path = os.path.join(data_dir, line)

                    data_size = int(os.path.getsize(img_path) / 4)
                    with open(img_path, "rb") as img_buf:
                        fmt_images = str(data_size) + 'f'
                        images = np.array((struct.unpack(fmt_images, img_buf.read(4 * data_size))), dtype=float)
                        images.reshape(input_shape)
                        full_images.append(images)
                        idx = idx + 1
        except Exception as e:
            print("Reader failed!\n{}".format(str(e)))
            os._exit(1)

        print("full images size: " + str(len(full_images)))

        offset = 0
        while True:
            if offset + buffer_size > len(full_images):
                break
            idx = offset
            offset = offset + buffer_size
            # quantization does not need label data, use random data for simplicity
            labels = np.random.random_integers(size=(buffer_size, ), low=0, high=9).astype('int64')
            for i in range(buffer_size):
                yield full_images[idx + i], int(labels[i])

    return reader