#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import json
import os
import random
import time

import requests

# 调用登录获取课程等基础配置
from src.baseinfo.base_info import BaseInfo
from src.baseinfo.base_policy import BasePolicy
from src.config.base_config import ReadBaseConfig

# 输出的日志文档
from src.config.config_logging import ConfigLogging


class PxEn:
    def __init__(self):
        self.basedata = ReadBaseConfig()
        # logger日志初始化
        self.logger = ConfigLogging().write_logging()
        # 获取服务端学生信息【0.3.3】
        self.info_data = BaseInfo()
        # 获取组件策略
        self.policy = BasePolicy()

    # 通用指令，倒计时组件【0.3.3】
    def handle_countdown(self, login_token, classid, lessonnum):
        self.logger.info("<<--------倒计时组件start----->>")
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": 20,
            "to": 30,
            "command": "COUNTDOWN_END",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {}
        headers["Content-type"] = "application/json"
        headers["T-px-Post-ID"] = str(random.randint(0, 10000))
        headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
        headers["T-px-Validate-Token"] = login_token
        headers["T-px-Lesson-Num"] = str(lessonnum)
        headers["T-px-Class-ID"] = classid
        headers["T-px-Client-Type"] = str(20)
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
            self.logger.info("handle_countdown:", result)
        except Exception as e:
            self.logger.error("loging error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("handle_countdown_ok:" + str(result))
        else:
            self.logger.info("handle_countdown_error:" + str(result))

    # 1V1组件启动【0.3.3】
    def en_onetoone(self, oto_list, login_token, classid, lessonnum):
        url = self.basedata.get_http("base_en_onetoone_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }  # ipad端发送请求
        data = oto_list
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
            self.logger.info("英语1V1点人：" + str(result))
        except Exception as e:
            self.logger.error("loging error:", e)

    # 1V1组件【AIB】
    def en_onetoone_aib(self, content, login_token, classid, lessonnum):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }  # ipad端发送请求
        data = {
            "from": "20",
            "to": "30",
            "command": "HAND_ANSWER_STU",
            "content": content,
            "requireHBCheck": True,
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("en_onetoone_aib_error:" + str(e))
        if result["code"] == 0:
            self.logger.info("en_onetoone_aib_true:" + str(result))
        else:
            self.logger.error("en_onetoone_aib_false:" + str(result))

    # 1V1问答【0.3.3】
    def en_handle_onetoone(
        self, msg, stuid_list, login_token, classid, lessonnum, command
    ):
        self.logger.info("<<---1V1问答_start--->>")
        msg["content"]["students"][0]["index"]
        msg["content"]["students"][0]["stuID"]
        msg["content"]["students"][0]["type"]
        msg["content"]["students"][0]["stuName"]
        _start_stu_score, _end_stu_score = 0, 0
        # 调用学生积分
        # stu_points1 = self.info_data.read_stu_point(stuid_list, login_token, classid, lessonnum)
        # for i in range(len(stu_points1)):
        #     if stu_points1[i]['stuID'] == student_id:
        #         self.logger.info("1V1环节开始前积分：" + str(student_name) + str(stu_points1[i]['rewardPoint']))
        #         start_stu_score = stu_points1[i]['rewardPoint']
        time.sleep(5)
        if command == "RECOGNIZE_HANDSUP_END":
            # 调用1V1问答组件模拟学生回答问题
            content = {
                "result": 1,  # 6 结束 7 提示 1 正确
            }
            policy = self.policy.read_excel(1, 1, 0, 7)
            self.logger.info("policy_count:" + str(policy))
            if policy and policy != 7:
                content["result"] = policy
            elif policy and policy == 7:
                content["result"] = policy
                self.en_onetoone_aib(
                    content, login_token, classid, lessonnum
                )  # n设置回答正确与否
                time.sleep(6)
                content["result"] = random.choice([1, 6])
            self.en_onetoone_aib(
                content, login_token, classid, lessonnum
            )  # n设置回答正确与否
        elif command == "RECOGNIZE_HANDSUP_END_PRO":  # 1V1pro
            result = 8
            policy = self.policy.read_excel(6, 2, 0, 9)
            self.logger.info("policy_count:" + str(policy))
            if policy and policy != 7:
                result = policy
            elif policy and policy == 7:
                result = random.choice([8, 9])
            self.info_data.get_transfor_api(
                login_token,
                classid,
                lessonnum,
                command="HAND_ANSWER_STU",
                content={"result": result},
            )
        # 1V1组件完成，调用查询学生积分
        # stu_points2 = self.info_data.read_stu_point(stuid_list, login_token, classid, lessonnum)
        # for i in range(len(stu_points2)):
        #     if stu_points2[i]['stuID'] == student_id:
        #         end_stu_score = stu_points2[i]['rewardPoint']
        # if end_stu_score <= start_stu_score:
        #     self.logger.error("1V1环节，学生添加积分异常,start_score:" + str(start_stu_score) + ",end_score:" + str(end_stu_score))

    #  看图展示组件
    def get_student_picture_list_api(
        self, login_token, classid, lessonnum, component_id
    ):
        self.logger.info(
            "<<---------------英语new看图展示组件 start------------------>>"
        )
        url = self.basedata.get_http("base_findStudentShowPictureList_url")
        result = None
        data = {"componentID": component_id}
        headers = {}
        headers["Content-type"] = "application/json"
        headers["T-px-Post-ID"] = str(random.randint(0, 10000))
        headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
        headers["T-px-Validate-Token"] = login_token
        headers["T-px-Lesson-Num"] = str(lessonnum)
        headers["T-px-Class-ID"] = classid
        headers["T-px-Client-Type"] = str(20)
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("loging error:", e)
        self.logger.info("获取看图展示学生列表：", result)
        stulist_showpicture = result["result"]["stuList"]
        return stulist_showpicture

    # 看图展示"REQUEST_SELECTTALKING_INFO"
    def get_request_selecttalking_info_api(
        self, login_token, classid, lessonnum
    ):
        self.logger.info(
            "<<-----------英语看图展示辅导端发送获取学生信息-------->>"
        )
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "REQUEST_SELECTTALKING_INFO",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {}
        headers["Content-type"] = "application/json"
        headers["T-px-Post-ID"] = str(random.randint(0, 10000))
        headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
        headers["T-px-Validate-Token"] = login_token
        headers["T-px-Lesson-Num"] = str(lessonnum)
        headers["T-px-Class-ID"] = classid
        headers["T-px-Client-Type"] = str(20)
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
            self.logger.info(
                "get_request_selecttalking_info_api_success：", result
            )
        except Exception as e:
            self.logger.error("get_request_selecttalking_info_api_error:", e)

    # 看图展示发送默认图片【0.3.3】
    def handle_showpicture_sendimage(
        self, stulist_sp, picture_urls, login_token, classid, lessonnum
    ):
        self.logger.info(
            "<<----------------英语看图展示发送默认图片------------------>>"
        )
        result = None
        if stulist_sp and picture_urls:
            stulist_sps = random.sample(stulist_sp, 1)  # 列表随机取一个学生
            picture_url = random.sample(picture_urls, 1)  # 随机选择默认图片
            url = self.basedata.get_http("base_transfer_url")
            data = {
                "from": "20",
                "to": "30",
                "command": "SELECTTALKING_SEND_URL",
                "content": {
                    "stuName": stulist_sps[0]["stuName"],
                    "stuID": stulist_sps[0]["stuID"],
                    "imgIndex": "0",  # -1表示上传默认图片
                    "imgUrl": picture_url,
                },
                "requireHBCheck": True,
            }
            headers = {}
            headers["Content-type"] = "application/json"
            headers["T-px-Post-ID"] = str(random.randint(0, 10000))
            headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
            headers["T-px-Validate-Token"] = login_token
            headers["T-px-Lesson-Num"] = str(lessonnum)
            headers["T-px-Class-ID"] = classid
            headers["T-px-Client-Type"] = str(20)
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                result = r.json()
            except Exception as e:
                self.logger.error("loging error:", e)
            self.logger.info("看图展示发送默认图片环节返回值：", result)
        else:
            self.logger.error(
                "看图展示发送默认图片环节获取学生名或imageurl error"
            )

    # 看图展示表扬点赞环节【0.3.3】
    def handle_showpicture_praise(self, login_token, classid, lessonnum):
        self.logger.info("看图展示表扬点赞环节 start")
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "SELECTTALKING_COMMEND",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {}
        headers["Content-type"] = "application/json"
        headers["T-px-Post-ID"] = str(random.randint(0, 10000))
        headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
        headers["T-px-Validate-Token"] = login_token
        headers["T-px-Lesson-Num"] = str(lessonnum)
        headers["T-px-Class-ID"] = classid
        headers["T-px-Client-Type"] = str(20)
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("handle_showpicture_sendimage_success:", e)
        self.logger.info("看图展示表扬点赞result：", result)
        return result["result"]["success"]

    # 看图展示点赞完成结束环节【0.3.3】
    def handle_showpicture_end(self, login_token, classid, lessonnum):
        self.logger.info("看图展示点赞完成")
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "SElECTTALKING_END",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {}
        headers["Content-type"] = "application/json"
        headers["T-px-Post-ID"] = str(random.randint(0, 10000))
        headers["T-px-Trace-ID"] = str(random.randint(0, 10000))
        headers["T-px-Validate-Token"] = login_token
        headers["T-px-Lesson-Num"] = str(lessonnum)
        headers["T-px-Class-ID"] = classid
        headers["T-px-Client-Type"] = str(30)
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("loging error:", e)
        self.logger.info("看图展示end：", result)

    # role play【0.4.0】
    def role_paly_repeat(self, login_token, classid, lessonnum, readindex):
        self.logger.info(" role_play_repeat_start")
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "to": "30",
            "command": "PLAY_EXAMPLE_AUDIO",
            "content": {"curLineIndex": readindex},
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Class-ID": classid,
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("rolepaly_repeat_error:", e)
        if result["code"] == 0:
            self.logger.info("rolepaly_repeat_true:", result)
        else:
            self.logger.error("rolepaly_repeat_false:", result)

    # magic hot【0.4.0】
    def magic_hat_stulist(self, login_token, classid, lessonnum, *args):
        self.logger.info(" 获取绑定学生信息列表")
        url = self.basedata.get_http("base_tag_url")
        result = None
        result_data = None
        data = {
            "strategy": args[2],
            "componentID": args[0],
            "componentType": args[1],
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Class-ID": classid,
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_tagstu_list_error:", e)
        if result["code"] == 0 and result["result"]:
            stu_lists = list(result["result"]["students"])
            for stu_list in stu_lists:
                if "tagType" in stu_list and int(stu_list["tagType"]) == 402:
                    stu_lists.remove(stu_list)  # 402 已答题
            result_data = stu_lists
        else:
            self.logger.error("get_tagstu_list_false:" + str(result))
        return result_data

    def magic_hot_next(self, login_token, classid, lessonnum, stuname, stuid):
        self.logger.info(
            "<<--------------------magic hot start------------------->>"
        )
        time.sleep(3)
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "MAGIC_HAT_PAD_NEXT",
            "content": {
                "stuName": stuname,
                "stuID": stuid,
                "result": "Excellent",
            },
            "requireHBCheck": True,
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Class-ID": classid,
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("magic_hot_next_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("magic_hot_next_true:", result)
        else:
            self.logger.error("magic_hot_next_false:", result)

    def magic_hot_end(self, login_token, classid, lessonnum):
        time.sleep(1)
        url = self.basedata.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "MAGIC_HAT_PAD_END",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Class-ID": classid,
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("magic_hot_end_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("magic_hot_end_true:", result)
        else:
            self.logger.error("magic_hot_end_false:", result)

    def magic_hot(self, msg, *args):
        strategy = "magic-hat"
        if msg["command"] == "MAGIC_HAT_END":
            self.magic_hot_end(args[0], args[1], args[2])
        else:
            tag_stulists = self.magic_hat_stulist(
                args[0], args[1], args[2], args[3], args[4], strategy
            )
            if len(tag_stulists) == 0:
                self.magic_hot_end(args[0], args[1], args[2])
            else:
                student_name = tag_stulists[0]["stuName"]
                student_id = tag_stulists[0]["stuID"]
                self.magic_hot_next(
                    args[0], args[1], args[2], student_name, student_id
                )


if __name__ == "__main__":
    data = PxEn()
    dicts = {
        "token": os.getenv("ALICE_AUTOTEST_WEBSOCKET_TOKEN", ""),
        "userid": os.getenv("ALICE_AUTOTEST_USER_ID", ""),
        "classid": os.getenv("ALICE_AUTOTEST_CLASS_ID", ""),
        "lessonnum": 1,
    }
    data.handle_showpicture(
        dicts["token"], dicts["classid"], dicts["lessonnum"]
    )
