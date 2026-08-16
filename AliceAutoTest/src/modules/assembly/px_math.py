#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import json
import os
import random
import time

import requests

from src.config.base_config import ReadBaseConfig

# 输出的日志文档
from src.config.config_logging import ConfigLogging


class PxMath:
    def __init__(self):
        self.basedata = ReadBaseConfig()
        # logger日志初始化
        self.logger = ConfigLogging().write_logging()

    # 数学1V1问答组件
    def px_onetoone(self, n, stuid_list, login_token, classid, lessonnum):
        stuids = []
        types_list = [1, 2]
        for i in range(len(stuid_list)):
            stuids.append(stuid_list[i]["stuID"])
        stuid = random.sample(stuids, 1)
        types = random.sample(types_list, 1)
        data = {
            "index": 1,  # 小题序号
            "result": 1,  # 1：正确 0：错误 2：无回答 3：再叫一人 4：提示话术 5：声音小话术
            "type": types[0],  # 1：普通 2：random
            "stuID": stuid[0],  # 学生ID
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

        url = self.basedata.get_http(
            "base_onetoone_url"
        )  # 获取1V1问答接口URL地址
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
        except Exception as e:
            print("loging error:", e)
        response_onetoone_datas = r.json()
        if response_onetoone_datas["result"]["success"]:
            print("1V1问答ipad端发送指令成功：", response_onetoone_datas)
            return stuid
        else:
            print("1V1问答发送指令失败")

    """
    做题投屏
    get_student_sign_list_api
    get_screen_question_api
    px_praiseset
    get_screen_question_celebration_api
    px_question_end
    question_answer
    """

    # 做题投屏-获取签到学生信息
    def get_student_sign_list_api(self, login_token, classid, lessonnum):
        stuid_list = []
        result = None
        url = self.basedata.get_http("base_findStudentSingList_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        data = {"sign": 1, "orderType": 2}
        try:
            rp = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = rp.json()
        except Exception as e:
            print("get_findStudentSignList_api:", e)
        if result["result"]["stuSignList"]:
            for signed_student in result["result"]["stuSignList"]:
                stuid_list.append(signed_student["stuID"])
        return stuid_list

    # 做题投屏-回答正确与否
    def get_screen_question_api(
        self, login_token, classid, lessonnum, isbind_sutid
    ):
        result_data = None
        result = None
        url = self.basedata.get_http("base_screenquestion_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        isbind = {
            "result": 1,
            "isLight": "",
            "stuID": isbind_sutid,
            # "answer": "A",
            # "answerTime": random.randint(0, 1000)
        }
        data = {
            "index": 1,
            "subType": 0,
            "componentID": "abc",
            "componentType": "",
            "requireHBCheck": True,
            "list": [isbind],
        }
        try:
            rp = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = rp.json()
        except Exception as e:
            self.logger.error("get_screenQuestion_api_error:", e)
        if result["code"] == 0 and result["result"]["success"]:
            result_data = isbind_sutid
        return result_data

    # 做题投屏-表扬复选框【0.4.0】
    def px_praiseset(self, login_token, classid, lessonnum):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "from": "20",
            "to": "30",
            "command": "ANSWERSCREEN_PRAISESET_STU",
            "content": {"praise": 1},
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
            self.logger.error("px_praiseset_error:", e)
        if result["code"] != 0:
            self.logger.error("px_praiseset_false:" + str(result))

    # 做题投屏-提前结束【0.4.0】
    def get_screen_question_celebration_api(
        self, login_token, classid, lessonnum, right_stuids
    ):
        result = None
        url = self.basedata.get_http("base_screenQuestionCelebrate_url")
        data = {"index": 1, "stuID": right_stuids[0], "type": 1, "imgUrl": ""}
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
            self.logger.error("get_screenQuestionCelebrate_api_error:", e)
        if result["code"] != 0:
            self.logger.error(
                "get_screenQuestionCelebrate_api_false:" + str(result)
            )

    # 做题投屏-提前结束【0.4.0】
    def px_question_end(self, login_token, classid, lessonnum):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "GROUP_QUESTION_ANSWER_END",
            "from": "20",
            "to": "30",
            "requireHBCheck": True,
            "content": {},
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
            print("px_question_end:", e)
        if result["code"] != 0:
            self.logger.error("px_question_end_false:" + str(result))

    # 做题投屏
    def question_answer(self, login_token, classid, lessonnum):
        """
        :param login_token:
        :param classid:
        :param lessonnum:
        :return:
        """
        isbind_stuids = self.get_student_sign_list_api(
            login_token, classid, lessonnum
        )
        if isbind_stuids:
            time.sleep(5)
            for isbind_stuid in isbind_stuids:
                self.get_screen_question_api(
                    login_token, classid, lessonnum, isbind_stuid
                )
            self.px_praiseset(login_token, classid, lessonnum)
            self.get_screen_question_celebration_api(
                login_token, classid, lessonnum, isbind_stuids
            )
            self.px_question_end(login_token, classid, lessonnum)

    """
    多人问答
    get_student_sign_details_api
    """

    # 获取学生签到详细列表
    def get_student_sign_details_api(self, login_token, classid, lessonnum):
        result = None
        url = self.basedata.get_http("base_findStudentSignDetailesList_url")
        data = {
            "componentID": 1,
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
            self.logger.error("get_findStudentSignDetailesList_api_error:", e)
        if result["code"] == 0 and result["result"]:
            return result["result"]["stuSignList"]
        else:
            self.logger.error(
                "get_findStudentSignDetailesList_api_false:" + str(result)
            )

    # 多人问答-选择学生回答
    def px_many_ask(self, login_token, classid, lessonnum, msg):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "CALLTHEROLL_RECALL",
            "from": "20",
            "to": "30",
            "componentID": msg["componentID"],
            "componentType": "",
            "content": {
                "stuName": msg["nextStudent"]["name"],
                "stuID": msg["nextStudent"]["stuID"],
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
            self.logger.error("px_many_ask_error:", e)
        if result["code"] != 0:
            self.logger.error("px_many_ask_false:" + str(result))

    # 多人问答-详讲略讲or提前结束
    def px_many_ask_end(self, login_token, classid, lessonnum, *args):
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "CALLTHEROLL_COMMON",
            "from": "20",
            "to": "30",
            "componentID": 1,
            "componentType": "",
            "content": {"status": args[0], "type": args[1]},
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
            response_datas = r.json()
            print("多人问答提前结束:", response_datas)
        except Exception as e:
            print("loging error:", e)

    # 多人问答
    def many_ask(self, login_token, classid, lessonnum, call_count, msg):
        signed_students = self.get_student_sign_details_api(
            login_token, classid, lessonnum
        )
        time.sleep(1)
        if msg["content"]["nowStudent"] and msg["content"]["nextStudent"]:
            self.px_many_ask(login_token, classid, lessonnum, msg["content"])
            call_count += 1
        time.sleep(5)
        if int(call_count) >= int(len(signed_students) - 1):
            self.px_many_ask_end(login_token, classid, lessonnum, "NO", 2)
            time.sleep(1)
            self.px_many_ask_end(login_token, classid, lessonnum, "YES", 3)
            call_count = 0
        return call_count

    """
    课后大表扬
    """

    # 读取老师最近一次的表扬人数
    def get_recent_praised_students(self, login_token, classid, lessonnum):
        url = self.basedata.get_http("base_getRecentSavePraiseStus_url")
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        data = {}
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            response_datas = r.json()
            print("读取老师最近一次表扬的人数:", response_datas)
        except Exception as e:
            print("loging error:", e)

    # 读取AI推荐名单
    def get_recommended_student_behavior(self, login_token, classid, lessonnum):
        url = self.basedata.get_http("base_getStuBehaviorRecommended_url")
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        data = {}
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            response_datas = r.json()
            print("AI智能推荐:", response_datas)
        except Exception as e:
            print("loging error:", e)

    # 获取学生正向行为类型
    def get_student_goal_by_type_id(
        self, login_token, classid, lessonnum, type_id
    ):
        student_id_list = []
        result = None
        url = self.basedata.get_http("base_getStuGoalByTypeID_url")
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        data = {"typeID": type_id}
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("px_getStuGoalByTypeID_error:" + str(e))
        if result["code"] == 0 and result["result"]["noCurrentGoalStu"]:
            for i in result["result"]["noCurrentGoalStu"]:
                student_id_list.append(i["stuID"])
            return student_id_list
        else:
            return None

    # 发放勋章
    def set_student_medals(
        self, login_token, classid, lessonnum, student_id_list
    ):
        behavior_ids = [1, 2, 3]
        medal_arguments = []
        result = None
        for behavior_id in behavior_ids:
            if len(student_id_list) > 3:
                for student_id in random.sample(student_id_list, 3):
                    medal_argument = {
                        "behaviorID": behavior_id,
                        "stuID": student_id,
                    }
                    medal_arguments.append(medal_argument)
            else:
                for student_id in student_id_list:
                    medal_argument = {
                        "behaviorID": behavior_id,
                        "stuID": student_id,
                    }
                    medal_arguments.append(medal_argument)
        url = self.basedata.get_http("base_setStuMedal_url")
        headers = {
            "content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": str(20),
        }
        data = {"stuMedalArgs": medal_arguments}
        try:
            r = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = r.json()
            self.logger.info("发放勋章指令：" + str(result))
        except Exception as e:
            self.logger.error("px_setStuMedal_error:" + str(e))

    # 点名问答-获取签到学生信息
    def get_student_sign_list_v2_api(self, login_token, classid, lessonnum):
        stuid_list = []
        result = None
        url = self.basedata.get_http("base_findStudentSingList_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        data = {"sign": 1, "isBind": 1}
        try:
            rp = requests.post(url=url, data=json.dumps(data), headers=headers)
            result = rp.json()
        except Exception as e:
            self.logger.error("get_findStudentSignList_api_error:", e)
        if result["result"]["stuSignList"]:
            for signed_student in result["result"]["stuSignList"]:
                stuid_list.append(signed_student["stuID"])
        return stuid_list

    # 点名问答
    def get_ma_onetoone_api(self, login_token, classid, lessonnum, stuid):
        result = None
        url = self.basedata.get_http("base_en_onetoone_url")
        headers = {
            "Content-type": "application/json",
            "T-px-Validate-Token": login_token,
            "T-px-Post-ID": str(random.randint(0, 10000)),
            "T-px-Trace-ID": str(random.randint(0, 10000)),
            "T-px-Class-ID": classid,
            "T-px-Lesson-Num": str(lessonnum),
            "T-px-Client-Type": "20",
        }
        data = {"index": 1, "result": 1, "typeID": "onetoone", "stuID": stuid}
        try:
            r = requests.post(url, data=json.dumps(data), headers=headers)
            result = r.json()
        except Exception as e:
            self.logger.error("get_ma_onetoone_api_error:", e)
        if result["code"] != 0:
            self.logger.error("get_ma_onetoone_api_false:" + str(result))

    """
    快速点名
    quick_qa_start
    quick_qa_end
    quick_qa
    """

    # 快速点名【数学】
    def quick_qa_start(self, login_token, classid, lessonnum, *args):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "QUICKRESULT",
            "from": "20",
            "to": "30",
            "componentID": 1,
            "componentType": args[0],
            "content": {
                "status": "right",
                "stuID": args[1],
                "stuName": args[2],
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
            self.logger.error("quick_qa_error:", e)
        if result["code"] != 0:
            self.logger.error("quick_qa_false:" + str(result))

    # 快速点名-提前结束【数学】
    def quick_qa_end(self, login_token, classid, lessonnum, *args):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "QUICKEND",
            "from": "20",
            "to": "30",
            "content": {},
            "componentType": args[0],
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
            self.logger.error("quick_qa_error:", e)
        if result["code"] != 0:
            self.logger.error("quick_qa_false:" + str(result))

    # 快速点名
    def quick_qa(self, login_token, classid, lessonnum, *args):
        self.quick_qa_start(
            login_token, classid, lessonnum, args[0], args[1], args[2]
        )
        for ma_stu_list in args[3]:
            time.sleep(3)
            self.quick_qa_start(
                login_token,
                classid,
                lessonnum,
                args[0],
                ma_stu_list["stuID"],
                ma_stu_list["stuName"],
            )
        time.sleep(1)
        self.quick_qa_end(login_token, classid, lessonnum, args[0])

    """
    彩蛋
    egg_play
    """

    # 彩蛋【数学】
    def egg_play(self, login_token, classid, lessonnum, *args):
        result = None
        url = self.basedata.get_http("base_transfer_url")
        data = {
            "command": "EGG_PLAY",
            "from": "20",
            "to": "30",
            "content": {"isPlay": 1},
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
            self.logger.error("egg_play_error:", e)
        if result["code"] != 0:
            self.logger.error("egg_play_false:" + str(result))


if __name__ == "__main__":
    token = os.getenv("ALICE_AUTOTEST_WEBSOCKET_TOKEN", "")
    classid = os.getenv("ALICE_AUTOTEST_CLASS_ID", "")
    lessonnum = "3"

    student_id_list = ["123123", "234234", "345345"]
    data = PxMath()
    data.set_student_medals(token, classid, lessonnum, student_id_list)
