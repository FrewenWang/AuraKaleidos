#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# 调用接口请求类request,
import json
import os
import random
import string
import time

import requests

from src.baseinfo.pc_info import BasePcInfo

# 获取配置文件中的基础信息
from src.config.base_config import GetLogin, GetVersion, ReadBaseConfig
from src.config.base_file import BaseFile
from src.config.config_ding import ConfigDing

# 调用logging日志接口
from src.config.config_logging import ConfigLogging
from src.config.config_mysql import MySqlHandler
from src.fileconfig.txt_file_config import TxtFileConfig
from src.utils.other.run_many_time import RunTimes

# 定义基础接口类


class BaseApi:
    def __init__(self):
        self.logger = ConfigLogging().write_logging()

    # 通用接口调用
    def base_api(self, url, method, headers, data):
        result = {}
        timeout = 6
        headers["content-type"] = "application/json"
        if method == "POST":
            try:
                rp = requests.post(
                    url=url,
                    data=json.dumps(data),
                    headers=headers,
                    timeout=timeout,
                )  # json格式需要使用json.dumps(data)
                result = rp.json()
            except BaseException as e:
                self.logger.error("base_api_post_error:" + str(e))
        elif method == "GET":
            try:
                rp = requests.get(
                    url=url, params=data, headers=headers, timeout=timeout
                )
                result = rp.json()
            except BaseException as e:
                self.logger.error("base_api_get_error:" + str(e))
        else:
            self.logger.error("base_api_method_error")
        return result


class BaseConfig:
    """
    * 文件加载初始化
    * 日志记录初始化
    * 钉钉推送初始化
    * QA中心基础信息

    """

    def __init__(self):
        self.readfile = TxtFileConfig()  # 读取文件
        self.ding = ConfigDing()
        # 获取学生端版本信息
        self.getversion = GetVersion()
        # 随机生成deviceId
        self.device_id = "".join(
            random.sample(string.ascii_letters + string.digits, 8)
        )
        # 获取QA课件信息文件
        self.status_info = self.readfile.read_file(
            r"C:\courseware_data\courseware_data.txt"
        )


# 定义一个获取所有基础信息的类接口
class BaseLogin(BaseApi, BaseConfig):
    """
    登录认证API
    """

    def __init__(self):
        BaseApi.__init__(self)
        BaseConfig.__init__(self)
        try:
            self.subject = self.status_info[4]  # 课件学科
        except Exception:
            self.subject = 1
        self.base_config_data = GetLogin()

    # ipad端登陆认证接口【0.4.0】
    def get_auth_api(self):
        result_data = {}
        url = self.base_config_data.get_http("base_auth_url")
        data = self.base_config_data.getlogin(self.subject)
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            # "T-px-Device-ID": 'auto_test_use',
            # "T-px-Client-Type": '20'
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=data)
        if rp["code"] == 0 and rp["result"]:
            result_data["token"] = rp["result"]["token"]
            result_data["username"] = data["username"]
            users = rp["result"]["users"]
            if users:
                for user in users:
                    if user["ename"] == self.base_config_data.get_login(
                        "ename"
                    ):
                        result_data["eid"] = user["eid"]
                        result_data["userid"] = user["userId"]
                        return result_data
                    else:
                        continue
            else:
                content = (
                    "auto当前" + data["username"] + "返回值users异常！！！"
                )
                self.ding.ding_test(2, content)
        elif int(rp["code"]) == 1101:
            content = "auto当前" + data["username"] + "用户名或密码错误！！！"
            self.ding.ding_test(2, content)
            return content
        else:
            content = "auto当前" + data["username"] + "返回值code码错误！！！"
            self.ding.ding_test(2, content)

    # offline_ipad端登陆认证接口【0.4.0】
    def offline_get_auth_api(self):
        result_data = {}
        url = self.base_config_data.get_http("base_auth_url")
        data = self.base_config_data.getlogin(self.subject)
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=data)
        if rp["code"] == 0 and rp["result"]:
            users = rp["result"]["users"]
            if users:
                result_data["username"] = data["username"]
                result_data["userid"] = users[0]["userId"]
                result_data["eid"] = users[0]["eid"]
                result_data["token"] = rp["result"]["token"]
            else:
                content = (
                    "auto当前" + data["username"] + "返回值users异常！！！"
                )
                self.ding.ding_test(2, content)
        elif int(rp["code"]) == 1101:
            content = "auto当前" + data["username"] + "用户名或密码错误！！！"
            self.ding.ding_test(2, content)
        else:
            content = "auto当前" + data["username"] + "返回值code码错误！！！"
            self.ding.ding_test(2, content)
        return result_data

    # ipad端登陆接口【0.4.0】
    def get_login_api(self):
        auth = self.get_auth_api()
        result_data = {"username": auth["username"]}
        url = self.base_config_data.get_http("base_login_url")
        data = {
            "client": "20",
            "token": auth["token"],
            "eId": auth["eid"],
            "userId": auth["userid"],
            "clientVersion": "auto_test_clientVersion",
            "channel": "PHOENIX",
            "deviceId": "phoenix_ipad_test_deviceId",
            "phone": os.getenv(
                "ALICE_AUTOTEST_PHONE",
                self.base_config_data.get_contacts("test_phone"),
            ),
        }
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Device-ID": "auto_test_use",
            "T-px-Client-Type": "20",
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=data)
        if rp["code"] == 0 and rp["result"]:
            result_data["token"] = rp["result"]["token"]
            result_data["userid"] = rp["result"]["user"]["userId"]
            result_data["eid"] = rp["result"]["user"]["eid"]
            self.logger.info("辅导端登录_SUCCESS:" + str(rp))
        else:
            self.logger.error("辅导端登录_ERROR:" + str(rp))
        return result_data

    # ipad端登陆接口【0.4.0】
    def offline_get_login_api(self):
        auth = self.offline_get_auth_api()
        result_data = {"username": auth["username"]}
        url = self.base_config_data.get_http("base_login_url")
        data = {
            "client": "20",
            "token": auth["token"],
            "eId": auth["eid"],
            "userId": auth["userid"],
            "clientVersion": "auto_test_clientVersion",
            "channel": "PHOENIX",
            "deviceId": "phoenix_ipad_test_deviceId",
            "phone": os.getenv(
                "ALICE_AUTOTEST_PHONE",
                self.base_config_data.get_contacts("test_phone"),
            ),
        }
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=data)
        if rp["code"] == 0 and rp["result"]:
            result_data["userid"] = rp["result"]["user"]["userId"]
            result_data["eid"] = rp["result"]["user"]["eid"]
            result_data["token"] = rp["result"]["token"]
            self.logger.info("辅导端登录_SUCCESS:" + str(rp))
        else:
            self.logger.error("辅导端登录_ERROR:" + str(rp))
        return result_data

    # ipad端获取登录教师信息
    def get_login_info_api(self, login_token):
        url = self.base_config_data.get_http("base_getLoginInfo_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {}
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 0 and rp["result"]:
            self.logger.info("get_getLoginInfo_url_true:" + str(rp))
        else:
            self.logger.error("get_getLoginInfo_url_flase:" + str(rp))

    # iPad端获取当天上课课程信息
    def get_current_day_lesson_list_api(self, login_token, teacher_id, *args):
        result_data = {}
        result = None
        url = self.base_config_data.get_http(
            "base_findCurrentDayLessonList_url"
        )
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": "",
            "T-px-Lesson-Num": "",
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {
            "teacherID": teacher_id,
        }
        try:
            rp = self.base_api(
                url=url, method="POST", headers=headers, data=datas
            )
            result = rp
        except Exception as e:
            self.logger.error(
                "get_findCurrentDayLessonList_api_error:" + str(e)
            )
        if result["code"] == 0 and result["result"]:
            class_info = result["result"]["classInfoList"][0]
            class_id = class_info["classID"]
            lesson_num = class_info["lessonNum"]
            result_data["content"] = class_info
            result_data["classID"] = class_id
            result_data["lessonNum"] = lesson_num
        else:
            self.logger.error(
                "get_findCurrentDayLessonList_api_false:" + str(result)
            )
        return result_data

    #  "GET_CUR_CLASSINFO"
    def get_cur_classinfo_url(self, login_token):
        url = self.base_config_data.get_http("base_transfer_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": "",
            "T-px-Lesson-Num": "",
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {
            "from": 20,
            "to": 30,
            "command": "GET_CUR_CLASSINFO",
            "content": {},
            "requireHBCheck": True,
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 0 and rp["result"]:
            self.logger.info("get_cur_classinfo_url_true:" + str(rp))
        else:
            self.logger.error("get_cur_classinfo_url_flase:" + str(rp))

    # iPad端登录
    def ipad_login(self):
        ipad_login_info = self.get_login_api()
        login_token = ipad_login_info["token"]
        login_userid = ipad_login_info["userid"]
        eid = ipad_login_info["eid"]
        username = ipad_login_info["username"]
        self.get_login_info_api(login_token)
        time.sleep(1)
        result = self.get_current_day_lesson_list_api(login_token, login_userid)
        result["token"] = login_token
        result["userid"] = login_userid
        result["eid"] = eid
        result["username"] = username
        # self.get_cur_classinfo_url(login_token)
        return result

    # iPad端登录
    def offline_ipad_login(self):
        ipad_login_info = self.offline_get_login_api()
        login_token = ipad_login_info["token"]
        login_userid = ipad_login_info["userid"]
        eid = ipad_login_info["eid"]
        username = ipad_login_info["username"]
        self.get_login_info_api(login_token)
        time.sleep(1)
        result = self.get_current_day_lesson_list_api(login_token, login_userid)
        result["token"] = login_token
        result["userid"] = login_userid
        result["eid"] = eid
        result["username"] = username
        return result

    # ipad端扫码登录学生端
    def scan_qr_code_api(self, login_token, qr_code):
        confirm_code = None
        url = self.base_config_data.get_http("base_scanQrCode_url")
        headers = {
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Client-Type": "20",
        }
        datas = {
            "deviceId": qr_code["deviceId"],
            "postId": qr_code["postId"],
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 1103:  # 检测超时
            # self.check_qr_code_api(login_token)
            self.logger.error("二维码超时_ERROR：" + str(rp))
        elif rp["code"] == 1105:  # 二维码过期
            # self.scan_qr_code_api(login_token, qr_code)
            self.logger.error("二维码过期_ERROR：" + str(rp))
        elif rp["code"] == 0:
            confirm_code = rp["result"]["confirmCode"]
        return confirm_code

    # ipad端确认学生端上课
    def confirm_qr_login_api(self, login_token, qr_code, confirm_code):
        url = self.base_config_data.get_http("base_checkQrLogin_url")
        headers = {
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Client-Type": "20",
        }
        datas = {
            "confirmCode": confirm_code,
            "deviceId": qr_code["deviceId"],
            "postId": qr_code["postId"],
            "isLogin": 1,
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] != 0:
            self.logger.error("确认PC端登录_ERROR：" + str(rp))
        else:
            self.logger.info("PC学生端登录_true:" + str(rp))

    # PC端登录,流程优化版本
    def pc_login(self, token):
        qr_code = BasePcInfo.get_pc_deviceid()  # 获取学生端二维码信息
        if qr_code:
            confirm_code = self.scan_qr_code_api(token, qr_code)  # 扫描登录
            if confirm_code:
                time.sleep(2)
                self.confirm_qr_login_api(
                    token, qr_code, confirm_code
                )  # 登录确认
        else:
            return False

    # 辅导端准备完成
    def get_initready_api(self, login_token, class_id, lesson_num):
        url = self.base_config_data.get_http("base_transfer_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lesson_num),
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {
            "from": 20,
            "to": 30,
            "command": "COMPONENTS_INIT_READY",
            "content": {},
            "requireHBCheck": True,
        }
        print(datas)
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        print("get_initready_api:", rp)

    # ipad端下课指令
    def end_lesson_api(self, login_token, classid, lessonnum):
        self.logger.info("------IPAD_LESSON_END------")
        result = None
        result_data = None
        url = self.base_config_data.get_http("base_offLesson_url")
        data = {}
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_offLesson_api_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("get_offLesson_api_true：" + str(result))
            result_data = True
        else:
            self.logger.error("get_offLesson_api_false：" + str(result))
        return result_data

    # ipad端下课指令
    def get_video_end_stu_api(
        self, login_token, classid, lessonnum, teacher_id
    ):
        self.logger.info("------IPAD_GetDict------")
        result = None
        url = self.base_config_data.get_http("base_transfer_url")
        data = {
            "from": 20,
            "to": 30,
            "command": "VIDEO_END_STU",
            "content": {"teacherID": teacher_id, "type": "INIT"},
            "requireHBCheck": True,
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_video_end_stu_api_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("iPad端发送学生端下课指令SUCCESS：", result)
        else:
            self.logger.error("iPad端发送学生端下课指令ERROR：", result)

    # @classmethod
    # def get_username(cls):
    #     # 获取QA课件信息文件
    #     status_info = cls().readfile.read_file(r'C:\courseware_data\courseware_data.txt')
    #     try:
    #         subject = status_info[4]
    #     except Exception as e:
    #         subject = 1
    #     username = cls().base_config_data.getlogin(subject)['username']
    #     return username


# 基础信息
class BaseInfo(BaseLogin):
    def __init__(self):
        BaseLogin.__init__(self)

    # 选择课程前SYNC_CLASS_BTN_STATUS
    def get_sync_class_btn_status_api(self, login_token, class_id, lesson_num):
        url = self.base_config_data.get_http("base_transfer_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lesson_num),
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {
            "from": 20,
            "to": 30,
            "command": "SYNC_CLASS_BTN_STATUS",
            "content": {"classID": class_id, "lessonNum": lesson_num},
            "requireHBCheck": True,
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 0:
            self.logger.info("get_sync_class_btn_status_url_true:" + str(rp))
        else:
            self.logger.error("************出现断点续播*****************")
            self.logger.error("get_sync_class_btn_status_url_false:" + str(rp))

    # ipad端获取进行中课程
    def get_active_lesson_list_api(
        self, login_token, class_id, lesson_num, userid
    ):
        url = self.base_config_data.get_http("base_getInLessonList_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lesson_num),
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {"teacherID": userid}
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 0:
            self.logger.info("get_getInLessonList_api_true:" + str(rp))
        else:
            self.logger.error("get_getInLessonList_api_false:" + str(rp))

    # iPad端初始化课程
    def initialize_lesson_api(self, login_token, class_id, lessonnum):
        result = None
        url = self.base_config_data.get_http("base_initialLesson_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        datas = {"cause": 0}
        try:
            rp = self.base_api(
                url=url, method="POST", headers=headers, data=datas
            )
            result = rp
            self.logger.info("get_intiialLesson_api_success：" + str(result))
        except Exception as e:
            self.logger.info("get_intiialLesson_api_error:", e)
        if result["code"] == 0 and result["result"]:
            result_data = True
        else:
            result_data = False
        return result_data

    # ipad端初始化操作
    def ipad_init(self, login_token, class_id, lesson_num, userid):
        self.get_sync_class_btn_status_api(login_token, class_id, lesson_num)
        self.get_active_lesson_list_api(
            login_token, class_id, lesson_num, userid
        )
        self.initialize_lesson_api(login_token, class_id, lesson_num)

    # iPad端获取学生列表信息
    def get_lesson_point_sum_api(self, login_token, class_id, lesson_num):
        result_data = {}
        url = self.base_config_data.get_http("base_findLessonPointSum_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lesson_num),
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {}
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["code"] == 0:
            result_data["pointList"] = rp["result"]["pointList"]
            self.logger.info("get_findLessonPointSum_api_true:" + str(rp))
        elif rp["code"] == 1302:
            self.logger.error("get_findLessonPointSum_api_false:" + str(rp))
        return result_data

    # iPad端选择课程
    def get_selectclass_api(self, login_token, class_id, lesson_num, content):
        result = None
        url = self.base_config_data.get_http("base_transfer_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": class_id,
            "T-px-Lesson-Num": str(lesson_num),
            "T-px-Client-Type": "20",  # ipad端
        }
        datas = {
            "from": 20,
            "to": 30,
            "command": "COURSE_DETAIL",
            "content": content,
            "requireHBCheck": True,
        }
        rp = self.base_api(url=url, method="POST", headers=headers, data=datas)
        if rp["result"]["success"]:
            self.logger.info("get_selectclass_api_true:" + str(rp))
        else:
            self.logger.error("get_selectclass_api_false:" + str(rp))
            return result

    # iPad端点击课程
    def ipad_clickclass(self, login_token, class_id, lesson_num, eid, content):
        contents = content
        point_list = self.get_lesson_point_sum_api(
            login_token, class_id, lesson_num
        )
        if point_list:
            contents["pointList"] = point_list["pointList"]
            contents["accountType"] = 4
            contents["eId"] = eid
            time.sleep(2)
            selectclass = self.get_selectclass_api(
                login_token, class_id, lesson_num, contents
            )
            if not selectclass:
                self.get_selectclass_api(
                    login_token, class_id, lesson_num, contents
                )
        else:
            self.logger.error("pointList_error:" + str(point_list))

    # 获取当前账号当前课次班级学生信息【0.4.0】
    def get_student_api(self, login_token, classid, lessonnum):
        result_data = {}
        url = self.base_config_data.get_http("base_get_student_info_url")
        headers = {
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        data = {}
        try:
            rp = requests.post(url, data=json.dumps(data), headers=headers)
            result_data = rp.json()
        except Exception as e:
            self.logger.info("loging error:", e)
        stuid_list = []  # 获取学生列表
        if result_data["code"] == 0 and result_data["result"]:
            stulist = result_data["result"]["stu"]  # 获取学生ID和学生姓名列表
            for i in stulist:
                stuid_list.append(i["stuID"])
        else:
            content = (
                "auto当前"
                + self.base_config_data.getlogin(self.subject)["username"]
                + "获取学生信息CODE码错误/返回学生列表信息失败！！！"
            )
            self.ding.ding_test(2, content)
        return stuid_list

    # 通用转发指令【0.4.0】
    def get_transfor_api(
        self, login_token, classid, lessonnum, command, content
    ):
        url = self.base_config_data.get_http("base_transfer_url")
        result = None
        if content:
            data = {
                "from": "20",
                "command": command,
                "to": "30",
                "requireHBCheck": "true",
                "content": content,
            }
        else:
            data = {"command": command, "to": "30", "content": ""}
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }  # 20表示iPad辅导端
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.info("loging error:", e)
        return result

    # ipad端获发起获取学生头像指令【0.3.2】
    def send_get_head(self, login_token, classid, lessonnum):
        url = self.base_config_data.get_http("base_get_head_url")
        result = None
        data = {}
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("loging error:", e)

        if result["result"]["success"]:
            self.logger.info("发送获取学生头像指令成功：", result)
        else:
            self.logger.error("ipad端发送获取头像指令失败：", result)

    # iPad端发送获取头像指令【0.3.3】
    def get_stu_headpictures_api(self, login_token, classid, lessonnum):
        url = self.base_config_data.get_http("base_transfer_url")
        result = None
        result_data = None
        datas = {
            "from": "20",
            "to": "30",
            "command": "GET_AVATARS",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            r = requests.post(url, data=json.dumps(datas), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_stu_headpictures_api_error:", e)
        if result["result"]["success"]:
            self.logger.info("辅导端发送获取头像指令成功：" + str(result))
            result_data = True
        else:
            self.logger.error("辅导端发送获取头像指令失败：" + str(result))
            result_data = False
        return result_data

    # iPad端获取头像图片gid
    def get_student_positions_api(self, login_token, classid, lessonnum):
        result = None
        gid_list = []
        url = self.base_config_data.get_http("base_findStuPositions_url")
        data = {}
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            rq = requests.post(url, data=json.dumps(data), headers=headers)
            result = rq.json()
        except Exception as e:
            self.logger.error("get_findStuPositions_api_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            student_positions = result["result"]["stuPositions"]
            self.logger.info("获取gid信息：" + str(student_positions))
            for student_position in student_positions:
                gid_stu_dict = {}
                gid_stu_dict["stuID"] = student_position["stuID"]
                gid_stu_dict["gid"] = student_position["gid"]
                gid_list.append(gid_stu_dict)
        else:
            self.logger.error("get_findStuPositions_api_false:", result)
        return gid_list

    # ipad端获取学生信息
    def get_student_sign_list_api(
        self, login_token, classid, lessonnum, target
    ):
        """
        * 获取需要绑定学生信息列表
        :param login_token:
        :param classid:
        :param lessonnum:
        :param target:
        :return:
        """
        url = self.base_config_data.get_http("base_findStudentSignList_url")
        result = None
        result_data = []
        data = {
            "sign": 2,  # 标记0表示全部状态的学生
            "orderType": 1,
        }
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            rp = requests.post(url, data=json.dumps(data), headers=headers)
            result = rp.json()
        except Exception as e:
            self.logger.error("get_findStudentSignList_api_exception:", e)
        if result["code"] == 0 and result["result"]:
            self.logger.info("get_findStudentSignList_api_true：" + str(result))
            unsigned_students = result["result"]["stuUnSignList"]
            signed_students = result["result"]["stuSignList"]
            if target == 1:
                if (
                    unsigned_students
                ):  # target 1表示获取未标记学生信息，2表示获取已标记学生信息
                    result_data = [
                        stuUnSignList["stuID"]
                        for stuUnSignList in unsigned_students
                    ]
                elif not unsigned_students and signed_students:
                    result_data = [
                        signed_students["stuID"]
                        for signed_students in signed_students
                    ]
            elif target == 2 and signed_students:
                result_data = [
                    signed_students["stuID"]
                    for signed_students in signed_students
                ]
        return result_data

    # iPad端发送绑定头像指令【0.3.3】
    def send_bind_headpictures(
        self, gid, stuid, login_token, classid, lessonnum
    ):
        bind_stu = []
        result = None
        if len(gid) > len(stuid):
            for i in range(len(stuid)):
                bind_stu_number = {}
                bind_stu_number["stuID"] = stuid[i]
                bind_stu_number["gid"] = gid[i]["gid"]
                bind_stu.append(bind_stu_number)
        else:
            for i in range(len(gid)):
                bind_stu_number = {}
                bind_stu_number["stuID"] = stuid[i]
                bind_stu_number["gid"] = gid[i]["gid"]
                bind_stu.append(bind_stu_number)
        bindstu = {"students": bind_stu}
        url = self.base_config_data.get_http("base_transfer_url")
        data = {
            "from": "20",
            "to": "30",
            "command": "BIND_AVATARS",
            "content": bindstu,
            "requireHBCheck": True,
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": login_token,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Class-ID": classid,
            "T-px-Client-Type": str(30),
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
            self.logger.info("send_bind_headpictures_success:" + str(bindstu))
        except Exception as e:
            self.logger.error("send_bind_headpictures_error:", e)
        if result["result"]["success"]:
            self.logger.info("发送绑定头像信息成功:" + str(result))
            result_data = bindstu
        else:
            self.logger.error("发送绑定头像信息失败:" + str(result))
            result_data = None
        return result_data

    # ipad端点击开始上课【0.4】
    def start_lesson_api(self, login_token, classid, lessonnum):
        self.logger.info("<<-------辅导端发起上课------>>")
        result = None
        url = self.base_config_data.get_http("base_startlesson_url")
        data = {}
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            rp = requests.post(url, data=json.dumps(data), headers=headers)
            result = rp.json()
        except Exception as e:
            self.logger.error("get_startLesson_api_error:", e)
        if result["code"] == 0:
            self.logger.info("get_startLesson_api_ok：" + str(result))
        else:
            self.logger.error("get_startLesson_api_error：" + str(result))

    # ipad通用转发指令"VIDEO_PLAY_STU" 修改断点续播逻辑
    def get_video_play_stu_api(
        self, login_token, classid, lessonnum, teacher_id, old_lesson_num
    ):
        url = self.base_config_data.get_http("base_transfer_url")
        result = None
        data = {
            "from": "20",
            "to": "30",
            "command": "VIDEO_PLAY_STU",
            "content": {
                "teacherID": teacher_id,
                "point": 160,
                "type": "INIT",
                "oldLessonNum": old_lesson_num,
            },
            "requireHBCheck": True,
        }
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_startLesson_api_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("get_startLesson_api_ok：", result)
        else:
            self.logger.error("get_startLesson_api", result)
        return result["code"]

    # 辅导端发起下课指令【0.3.3】
    def handle_class_end(self, login_token, classid, lessonnum):
        self.logger.info(
            "<<--------------------辅导端发起下课指令---------------------------->>"
        )
        url = self.base_config_data.get_http("base_sign_img_stu_url")
        result = None
        data = {"to": "30", "command": "VIDEO_END_STU", "content": {}}
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
            self.logger.error("api loging error:", e)
        self.logger.info("辅导端发起下课指令数据返回值：", result)

    # 课前检测
    def get_checkaction_url(self, login_token, classid, lessonnum):
        url = self.base_config_data.get_http("base_transfer_url")
        result = None
        data = {
            "from": 20,
            "to": 30,
            "command": "TOTAL_RECHECK",
            "content": {},
            "requireHBCheck": True,
        }
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_checkaction_url_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            self.logger.info("get_checkaction_url_ok:" + str(result))
        else:
            self.logger.info("get_checkaction_url_error:" + str(result))

    # 读取学生积分
    def read_stu_point(self, stuid_list, login_token, classid, lessonnum):
        self.logger.info("学生ID列表信息：" + str(stuid_list))
        url = self.base_config_data.get_http("base_read_points_url")
        result = None
        data = {"stuIDs": stuid_list}
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        if data["stuIDs"]:
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                result = r.json()
            except Exception as e:
                self.logger.error("read_stu_point:", e)
            if result["result"]["success"]:
                return result["result"]["studentPointList"]
            else:
                self.logger.error("read_stu_point:", result)
                return 0
        else:
            self.logger.error("read_stu_point_stuid_list_error:", stuid_list)

    # 统计课件播放过程中组件数量【暂时未使用】
    def cp_count(self, msg, cp_list):
        if (
            msg["command"] == "SCHEDULE_UPDATE"
            and msg["content"]["components"] != ""
        ):
            components_list = msg["content"]["components"]
            for i in range(len(components_list)):
                if components_list[i]:
                    if components_list[i]["status"] == "end":
                        cp_list.append(components_list[i]["name"])
                else:
                    continue
        return cp_list

    # 统计列表中元素个数【暂时未使用】
    def count_number(self, cp_list):
        assembly_count = {}
        myset = set(cp_list)
        for item in myset:
            assembly_count[item] = cp_list.count(item)
        return assembly_count

    # 获取发放积分总分
    def get_lesson_points_api(self, login_token, classid, lessonnum):
        url = self.base_config_data.get_http("base_findLessonPoints_url")
        result = None
        result_data = None
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        data = {}
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_findLessonPoints_api_error:", e)
        if result["result"]["success"]:
            result_data = result["result"]["leftTutorPoint"]
        return result_data

    # 获取班级积分【发放积分使用】
    def change_points_api(
        self,
        login_token,
        classid,
        lessonnum,
        isbind_stuids,
        remaining_tutor_points,
    ):
        detail_list = []
        result = None
        if isbind_stuids and remaining_tutor_points >= len(isbind_stuids) * 5:
            for i in range(random.randint(1, len(isbind_stuids))):
                detail = {
                    "changeID": i,
                    "opsType": 0,
                    "bizType": 10,
                    "bizExtendProperties": "测试1",
                    "stuID": isbind_stuids[i],
                    "opsPoints": 2,
                }
                detail_list.append(detail)
        url = self.base_config_data.get_http("base_pointChange_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        data = {"detailList": detail_list}
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_pointChange_api_error:", e)
        if result["code"] != 0:
            self.logger.info("get_pointChange_api_false：" + str(result))

    def point_change(self, login_token, classid, lessonnum):
        remaining_tutor_points = self.get_lesson_points_api(
            login_token, classid, lessonnum
        )
        isbind_stuids = self.get_student_sign_list_api(
            login_token, classid, lessonnum, 2
        )
        self.change_points_api(
            login_token,
            classid,
            lessonnum,
            isbind_stuids,
            remaining_tutor_points,
        )

    # 获取绑定学生信息列表【0.4.0】
    def get_tagstu_list(self, login_token, classid, lessonnum, *args):
        self.logger.info(" 获取绑定学生信息列表")
        url = self.base_config_data.get_http("base_tag_url")
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

    # # 课中信息交互
    # def get_config_api(self, login_token, classid, lessonnum, type):
    #     result = None
    #     url = self.base_config_data.get_http('base_getConfig_url')
    #     data = {
    #         "type": type
    #     }
    #     headers = {
    #         'Content-type': 'application/json',
    #         'T-px-Validate-Token': login_token,
    #         'T-px-Post-ID': str(random.randint(0, 10000)),
    #         'T-px-Trace-ID': str(random.randint(0, 10000)),
    #         'T-px-Class-ID': classid,
    #         'T-px-Lesson-Num': str(lessonnum),
    #         'T-px-Client-Type': '20'
    #     }
    #     try:
    #         rp = self.base_api(url=url, method='GET', headers=headers, data=data)
    #         result = rp
    #         print("getConfig:", result)
    #     except Exception as e:
    #         self.logger.error("get_getConfig_api_error:", e)
    #     if result["code"] == 0 and result['result']['success'] == True:
    #         self.logger.info("get_getConfig_api_ok:" + str(result))
    #     else:
    #         self.logger.info("get_getConfig_api_error:" + str(result))
    #
    # # 课中信息交互1
    # def getConfig(self, login_token, classid, lessonnum):
    #     type1 = "hallOfFame"
    #     self.get_config_api(login_token, classid, lessonnum, type1)
    #     type2 = "listOnTheWall"
    #     self.get_config_api(login_token, classid, lessonnum, type2)


# QA中心同步测试状态数据
class BaseResult(BaseConfig):
    """
    * 用于同步课件校验信息

    3 课件状态
    d6be44aaccc44de4874b71298f99853d 课件ID
    3 课件版本
    null 所需要端版本
    2 学课编号，1数学，2英语
    10 跑课总次数
    """

    def __init__(self):
        BaseConfig.__init__(self)
        self.logger = ConfigLogging().write_logging()
        self.base_config_data = ReadBaseConfig()
        self.path = r"\px_pt_auto\BaseConfig\checksum.txt"
        self.basefile = BaseFile()

    def test_status(self, status_code, content):
        """
        课件状态同步
        :param status_code: 1 待测试 2 下载中 3 测试中 4 测试通过 5 测试不通过 6 失效
        :param content:
        :return:
        """
        status_info = self.status_info
        url = self.base_config_data.get_http("test_status_url")
        result = None
        headers = {
            "content-type": "application/json",
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": "sss",
        }
        parm_data = {
            "coursewareId": None,
            "coursewareVersion": None,
            "checkStatus": status_code,
            "checkResult": None,
        }
        if status_info and len(status_info) >= 3:
            parm_data["coursewareId"] = status_info[1]  # 课件ID
            parm_data["coursewareVersion"] = int(status_info[2])  # 课件版本

        if int(status_code) in [4, 5] and content:
            parm_data["checkResult"] = {
                "errorReason": " ",
                "clientVersion": content["px_version"],
                "computerName": content["hostname"],
                "macAddress": content["mac"],
                "testAccount": content["username"],
                "lastVideoId": content["cid"],
                "componentStatistics": str(content["componentcount"]),
            }
        elif int(status_code) in [4, 5] and not content:
            self.logger.error("content数据存在问题")
        try:
            rp = requests.post(url, data=json.dumps(parm_data), headers=headers)
            result = rp.json()
            self.logger.info(result)
        except Exception as e:
            self.logger.error("test_status_loging_error:", e)

    # 获取QA课件校验状态信息
    def test_checkresult(self):
        status_info = self.status_info
        url = self.base_config_data.get_http("test_check_url")
        headers = {
            "content-type": "application/json",
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": "sss",
        }
        parm_data = {"coursewareId": None, "coursewareVersion": None}
        if status_info and len(status_info) >= 3:
            parm_data["coursewareId"] = status_info[1]
            parm_data["coursewareVersion"] = int(status_info[2])
        try:
            rp = requests.post(url, data=json.dumps(parm_data), headers=headers)
            result_data = rp.json()
            return len(result_data["result"]["coursewareCheckResultDTOS"])
        except Exception as e:
            self.logger.error("test_checkresult:", e)

    def test_pretest(self, status_code, content):
        """
        预测试状态信息同步
        :param status_code: # 1 测试通过 2 测试不通过
        :param content:
        :return:
        """
        status_info = self.status_info
        url = self.base_config_data.get_http("test_pretest_url")
        headers = {
            "content-type": "application/json",
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Validate-Token": "sss",
        }
        parm_data = {
            "coursewareId": None,
            "coursewareVersion": None,
            "checkBeginDate": int(content["checkstarttime"] * 1000),
            "checkEndDate": int(content["checkendtime"] * 1000),
            "checkStatus": status_code,
            "clientVersion": None,
            "macAddress": None,
            "computerName": None,
            "errorReason": None,
            "testAccount": None,
            "lastVideoId": None,
            "componentStatistics": None,
        }
        if status_info and len(status_info) >= 3:
            parm_data["coursewareId"] = status_info[1]
            parm_data["coursewareVersion"] = int(status_info[2])
        if int(status_code) == 1 and content:
            parm_data["clientVersion"] = content["px_version"]
            parm_data["macAddress"] = content["mac"]
            parm_data["computerName"] = content["hostname"]
            parm_data["errorReason"] = content["errorReason"]
            parm_data["testAccount"] = content["username"]
            parm_data["lastVideoId"] = content["cid"]
            parm_data["componentStatistics"] = str(content["componentcount"])
        try:
            rp = requests.post(url, data=json.dumps(parm_data), headers=headers)
            result_data = rp.json()
            return result_data
        except Exception as e:
            self.logger.error("test_pretest_loging_error:" + str(e))

    def get_classname(self):
        """
        获取校验课件name
        :return:
        """
        classname = self.base_config_data.get_login("classname")  # 获取配置
        if int(self.base_config_data.get_login("username_status")) not in [
            1,
            2,
        ]:
            # 获取QA课件信息文件
            courseware_id = self.status_info[1]
            courseware_version = self.status_info[2]
            classname = str(courseware_id) + ">>" + str(courseware_version)
        return classname

    def get_courseware_data(self):
        """
        获取自动化跑课校验次数
        :return:
        """
        courseware_data = self.status_info[5]  # 课程校验次数
        check_result_data = self.test_checkresult()
        return bool(
            courseware_data and int(check_result_data) >= int(courseware_data)
        )

    def update_checksum(self):
        with open(self.path, "a", encoding="utf-8") as file:
            file.write("1")
            file.close()

    def delete_checksum(self):
        with open(self.path, "r+", encoding="utf-8") as file:
            file.read()
            file.seek(0)
            file.truncate()
            file.close()

    def test_result(self, content, state):
        """
        测试结果同步
        :param content:
        :param state1: 1:测试通过 2:测试不通过
        :param state2: 1:测试通过 2:测试不通过 3:课前检测异常
        :param state3: 1 待测试 2 下载中 3 测试中 4 测试通过 5 测试不通过 6 失效
        :return:
        """
        mq = MySqlHandler()
        self.test_pretest(state, content)  # 同步预测试状态 state1
        self.logger.info(
            "同步预测试状态" + str(content["result_name"])
        )  # 统计记录日志
        self.ding.ding_test(state, content)  # 钉钉推送 state2
        try:
            mq.handle_insert(content)  # 数据状态同步数据库
        except Exception as e:
            self.logger.error("mysql error:", str(e))
        # gc = self.get_courseware_data()  # 检测是否需要重复跑课                              #去除本地多次跑课
        if int(state) == 2:
            self.test_status(5, content)  # 同步QA中心最终状态 state3
            self.logger.info(
                "同步最终结果到课件中心：测试不通过" + str(content)
            )  # 统计记录日志
            print("同步最终结果到课件中心：测试不通过" + str(content))
            self.basefile.all_data()  # 清空所有课件相关数据
        # elif int(state) == 1 and gc:                                                      #去除本地多次跑课
        elif int(state) == 1:  # 新增多节点机跑课，判断测试通过
            # self.test_status(4, content)  # 同步QA中心最终状态 state3                      #去除本地多次跑课
            runtime = RunTimes(content=content)
            runtime.test_runtime()
            self.basefile.all_data()  # 清空所有课件相关数据
        else:
            self.basefile.update_pid("0")  # 更新自动化循环检测状态


if __name__ == "__main__":
    content3 = {
        "title": "auto-课件自动化测试报告",
        "hostname": "NODE_662",
        "mac": "00-01-2E-94-9F-F8",
        "px_version": "2.1.0.312_release",
        "username": "offlan7",
        "classid": "ace>>3",
        "classname": "ace>>3",
        "cid": "a0924ab9bba040f2abafb13f8a4932d1",
        "error_code": 100,
        "error_detail": "无",
        "errorReason": "无",
        "componentcount": {
            "en_grab-red-packet-bird_1.0.6": 1,
            "en_parent-said_1.0.6": 1,
            "en_answer-machine_1.0.8": 8,
            "en_countdown_1.0.5": 5,
            "en_picture-talk-show_1.0.7": 1,
            "en_group-follow-up_1.0.7": 9,
            "en_state-volume-pk_1.0.6": 8,
            "en_one-to-one-handsup_1.0.9": 6,
            "en_light-follow-up-avatar-praise_1.0.1": 10,
            "en_one-to-one-point-pro_1.0.7": 1,
            "en_state-divide-group_1.0.8": 4,
        },
        "used_time": "2413S",
        "checkstarttime": 1592560037,
        "checkendtime": 1592562451,
        "remote_id": "144 485 834",
        "remote_pwd": "123456",
        "result_code": 0,
        "result_name": "【测试通过】",
    }
    content4 = {
        "title": "auto-课件自动化测试报告",
        "hostname": "NODE_662",
        "mac": "00-01-2E-94-9F-F8",
        "px_version": "2.1.0.312_release",
        "username": "offlan7",
        "classid": "61255642be014a67b5468aebbd7b95ec>>5",
        "classname": "61255642be014a67b5468aebbd7b95ec>>5",
        "cid": "888c11b37e754f839a9f3bd0134dfcc4",
        "error_code": 100,
        "error_detail": "无",
        "errorReason": "无",
        "componentcount": {
            "en_one-to-one-handsup-pro_1.0.7": 12,
            "en_light-follow-up-avatar_1.0.8": 1,
            "en_answer-machine_1.0.8": 9,
            "en_countdown_1.0.5": 6,
            "en_picture-talk-show_1.0.7": 1,
        },
        "used_time": "5523S",
        "checkstarttime": 1592986124,
        "checkendtime": 1592991647,
        "remote_id": "144 485 834",
        "remote_pwd": "123456",
        "result_code": 0,
        "result_name": "【测试通过】",
    }
    data = BaseResult()
    data.test_result(content3, 1)
    # ipad_login = data.ipad_login()
    # # 获取ipad端登录数据
    # login_token = ipad_login['token']
    # print(login_token)
    # login_userid = ipad_login['userid']
    # print(login_userid)
    # classid = ipad_login['classID']
    # print(classid)
    # lessonnum = ipad_login['lessonNum']
    # eid = ipad_login['eid']
    # content = ipad_login["content"]
    # username = ipad_login["username"]
