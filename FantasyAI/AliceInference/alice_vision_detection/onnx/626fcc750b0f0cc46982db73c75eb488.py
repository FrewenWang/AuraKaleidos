#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@Desc     : 模板匹配
@Author   : Zhang Meijun
Data      : 2019/2/11
"""
import cv2
import os


class Templates(object):
    """模版匹配"""

    def createTemplate(self, frame, points):
        """
        保存模板(脸部)
        :param points:106人脸关键点
        :return:
        """
        cp_Xmin = points[0][0]
        cp_Xmax = points[0][0]
        cp_Ymin = points[0][1]
        cp_Ymax = points[0][1]
        for cp in range(len(points)):  # 一张人脸的点最大最小值
            if points[cp][0] > cp_Xmax:
                cp_Xmax = points[cp][0]
            if points[cp][0] < cp_Xmin:
                cp_Xmin = points[cp][0]
            if points[cp][1] > cp_Ymax:
                cp_Ymax = points[cp][1]
            if points[cp][1] < cp_Ymin:
                cp_Ymin = points[cp][1]
        # print cp_Xmax, cp_Xmin, cp_Ymax, cp_Ymin
        width = cp_Xmax - cp_Xmin
        height = cp_Ymax - cp_Ymin
        rect_size = max(width, height)
        centerx = int(cp_Xmin + (width / 2.0))
        centery = int(cp_Ymin + (height / 2.0))
        # centerx = int((cp_Xmax + cp_Xmin) / 2)
        # centery = int((cp_Ymax + cp_Ymin) / 2)

        # 车内 * 1.2
        template_image = frame[centery - int(rect_size * 1.2 / 2.):centery + int(rect_size * 1.2 / 2.),
                         centerx - int(rect_size * 1.2 / 2.):centerx + int(rect_size * 1.2 / 2.)]

        # 室内
        # template_image = frame[centery - int(rect_size / 2.):centery + int(rect_size / 2.),
        #                  centerx - int(rect_size / 2.):centerx + int(rect_size / 2.)]

        # cv2.imwrite("./temp0.jpg", template_image)
        # print "template_image", template_image
        return template_image

    def createTemplateForHand(self, frame, handOneRect):
        """
        保存模板(手部)
        :param handOneRect:手势框
        :return:
        """
        cp_Xmin = min(handOneRect[0], handOneRect[2])
        cp_Xmax = max(handOneRect[0], handOneRect[2])
        cp_Ymin = min(handOneRect[1], handOneRect[3])
        cp_Ymax = max(handOneRect[1], handOneRect[3])

        width = cp_Xmax - cp_Xmin
        height = cp_Ymax - cp_Ymin
        rect_size = max(width, height)
        centerx = int(cp_Xmin + (width / 2.0))
        centery = int(cp_Ymin + (height / 2.0))
        # centerx = int((cp_Xmax + cp_Xmin) / 2)
        # centery = int((cp_Ymax + cp_Ymin) / 2)
        # print "centerx, centery", centerx, centery

        # # 车内 * 1.2
        template_image = frame[centery - int(rect_size * 1.2 / 2.2):centery + int(rect_size * 1.2 / 2.2),
                         centerx - int(rect_size * 1.2 / 4.2):centerx + int(rect_size * 1.2 / 4.2)]

        # 室内
        # template_image = frame[centery - int(rect_size / 2.):centery + int(rect_size / 2.),
        #                  centerx - int(rect_size / 3.5):centerx + int(rect_size / 3.5)]

        cv2.imwrite("./temp0.jpg", template_image)
        return template_image

    def useTemplate(self, frame, template):
        """
        人脸模板匹配
        :param frame: 图片
        :return: 人脸矩形框
        """
        if template.shape[0] > 0 and template.shape[1] > 0:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (1280, 720)
            try:
                img_gray = cv2.resize(img_gray,
                                      (img_gray.shape[1] / 4, img_gray.shape[0] / 4))  # 1/4*1/4  (320, 180)
                template = cv2.resize(template,
                                      (template.shape[1] / 4, template.shape[0] / 4))
                w, h = template.shape[::-1]  # 模板宽/4,高/4
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # 模板匹配
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc  # 左上角的点
                bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角的点
                # faceRectsFinal.append([top_left[0] * 4, top_left[1] * 4, bottom_right[0] * 4, bottom_right[1] * 4])
                # cv2.rectangle(frame, (int(top_left[0] * 4), int(top_left[1] * 4)),
                #               (int(bottom_right[0] * 4), int(bottom_right[1] * 4)),
                #               (0, 255, 255), 2)
            except:
                return []
            return [top_left[0] * 4, top_left[1] * 4, bottom_right[0] * 4, bottom_right[1] * 4]
        else:
            return []

    def handUseTemplate(self, frame, template):
        """
        手势模板匹配
        :param frame: 图片
        :return: 人脸矩形框
        """
        if template.shape[0] > 0 and template.shape[1] > 0:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (1280, 720)
            try:
                img_gray = cv2.resize(img_gray,
                                      (img_gray.shape[1] / 4, img_gray.shape[0] / 4))  # 1/4*1/4  (320, 180)
                template = cv2.resize(template,
                                      (template.shape[1] / 4, template.shape[0] / 4))
                w, h = template.shape[::-1]  # 模板宽/4,高/4
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # 模板匹配
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc  # 左上角的点
                bottom_right = (top_left[0] + w, top_left[1] + h)  # 右下角的点
                # faceRectsFinal.append([top_left[0] * 4, top_left[1] * 4, bottom_right[0] * 4, bottom_right[1] * 4])
                # cv2.rectangle(frame, (int(top_left[0] * 4), int(top_left[1] * 4)),
                #               (int(bottom_right[0] * 4), int(bottom_right[1] * 4)),
                #               (0, 255, 255), 2)
            except:
                return []
            return [top_left[0] * 4, top_left[1] * 4, bottom_right[0] * 4, bottom_right[1] * 4]
        else:
            return []


    #-----------------------------------------------------------

    # def createTemplate(self, frame, points):
    #     """
    #     保存模板(脸部)
    #     :param points:106人脸关键点
    #     :return:
    #     """
        # cp_Xmin = points[0][0]
        # cp_Xmax = points[0][0]
        # cp_Ymin = points[0][1]
        # cp_Ymax = points[0][1]
        # for cp in range(len(points)):  # 一张人脸的点最大最小值
        #     if points[cp][0] > cp_Xmax:
        #         cp_Xmax = points[cp][0]
        #     if points[cp][0] < cp_Xmin:
        #         cp_Xmin = points[cp][0]
        #     if points[cp][1] > cp_Ymax:
        #         cp_Ymax = points[cp][1]
        #     if points[cp][1] < cp_Ymin:
        #         cp_Ymin = points[cp][1]
        # print cp_Xmax, cp_Xmin, cp_Ymax, cp_Ymin
        # width = cp_Xmax - cp_Xmin
        # height = cp_Ymax - cp_Ymin
        # delta = max(width, height)
        # # center_x = (cp_Xmax - cp_Xmin) / 2 + cp_Xmin
        # # center_y = (cp_Ymax - cp_Ymin) / 2 + cp_Ymin
        #
        # if delta == width:  # 宽大于高
        #     cp_Ymin = cp_Ymin - (width - height) / 2
        #     cp_Ymax = cp_Ymax + (width - height) / 2
        # else:
        #     cp_Xmin = cp_Xmin - (height - width) / 2
        #     cp_Xmax = cp_Xmax + (height - width) / 2
        #
        # template_image = frame[cp_Ymin - int(delta * 0.02):cp_Ymax + int(delta * 0.02),
        #                  cp_Xmin - int(delta * 0.02):cp_Xmax + int(delta * 0.02)]
        #
        # # cv2.imwrite("./temp0.jpg", template_image)
        # return template_image

    # def template_detect(self):
    #     """
    #     检查是否存在模板
    #     """
    #     template = []
    #
    #     files = os.listdir("./templates")
    #     for file in files:
    #         if file.startswith("temp"):
    #             template.append(file)  # 模板列表
    #
    #     if len(template) > 0:
    #         for temp in template:
    #             temp_path = "./templates/" + temp
    #             template = cv2.imread(temp_path, 0)
    #             if template is not None:
    #                 return True, template
    #             else:
    #                 return False, template
