
import cv2
import numpy as np


def judging_camera_block(img, cam_type):
    """
    Main funtion of algorithm
    :param img: input image
    :param cam_type: 'oms_ir', 'oms_rgb', 'dms'.
    :return: if_block, block_score, block_ratio
    """
    # Trans to gray scale and resize
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # For RGB image: 3 channels
    else:
        img_gray = img  # For IR image: single channel
    target_size = (100, 100)
    img_gray = cv2.resize(img_gray, dsize=target_size, interpolation=cv2.INTER_AREA)

    # Judging algorithms
    img_gray_np = np.array(img_gray, dtype=float)
    img_gaussblur_np = cv2.GaussianBlur(img_gray_np, ksize=(3, 3), sigmaX=1000.0)
    delta_img = abs(img_gray_np - img_gaussblur_np)

    block_score = np.mean(delta_img)
    is_block = False
    block_ratio = 0  # used in dms camera
    if cam_type == 'oms_ir':
        if block_score < 2.45:
            is_block = True
    elif cam_type == 'oms_rgb':
        if block_score < 5.50:
            is_block = True
    elif cam_type == 'dms':
        grid_num = 15
        thresh_hold_multizone = 1.5
        thresh_grid_ratio = 0.71
        block_score_list = []  # block score for multi zones
        block_num = 0
        grid_width = int(target_size[0] / grid_num)
        grid_height = int(target_size[1] / grid_num)
        center_block_num = 0
        center_inds_up = 6
        center_inds_left = 4
        for i in range(grid_num):
            interval_h_start = i * grid_height
            interval_h_end = (i + 1) * grid_height
            for j in range(grid_num):
                interval_w_start = j * grid_width
                interval_w_end = (j + 1) * grid_width
                delta_block = np.mean(delta_img[interval_h_start: interval_h_end, interval_w_start: interval_w_end])
                if delta_block < thresh_hold_multizone:
                    block_num += 1
                block_score_list.append(delta_block)
                if i >= center_inds_up and j >= center_inds_left:
                    if delta_block < thresh_hold_multizone:
                        center_block_num += 1
        block_ratio = float(center_block_num / ((grid_num - center_inds_up) * (grid_num - center_inds_left)))  # ratio of blocked zones

        # Judge whether block
        if block_ratio >= thresh_grid_ratio:
            is_block = True  # The camera is blocked
            up = np.mean(img_gray[int(0.45 * target_size[1]): int(0.60 * target_size[1]),
                         int(0.4 * target_size[0]): int(0.6 * target_size[0])])
            down = np.mean(img_gray[int(0.8 * target_size[1]): int(0.93 * target_size[1]),
                           int(0.4 * target_size[0]): int(0.6 * target_size[0])])
            if down > 100 and down / up > 10:  # avoid no driver recognized occluded
                is_block = False
        else:
            is_block = False  # The camera is unblocked
    return is_block, block_score, block_ratio


if __name__ == '__main__':
    image_path = './dms_occlude.jpg'
    img = cv2.imread(image_path)
    result = judging_camera_block(img, 'dms')
    print(result)

