#!/usr/bin/env python
# @Time    : 2019/7/24 11:45
# @Author  : liuqi
# @Site    :
# @File    : px_run.py
# @Software: PyCharm
# 获取配置文件中的基础信息
# 新增版本号统计
import multiprocessing
import os
import random
import string
import time
from multiprocessing import Pipe, Process

from src.config.base_config import GetVersion, ReadBaseConfig
from src.config.base_file import BaseFile
from src.config.config_ding import ConfigDing
from src.config.config_hardware import HardwareConfig, PidConfig
from src.config.config_logging import ConfigLogging
from src.config.config_screen import ConfigScreen

# 调用超过三个小时报错策略
from src.modules.tools.three_hours import main

# 调用版本号统计方法
from src.utils.other.get_com_version import ComVersion

# 启动虚拟视频播放器
try:
    from tools.VCamTestTool.control2 import run_vcamtest
except ImportError:

    def run_vcamtest(*args, **kwargs):
        pass

# 调用模拟websocket长连接函数

# 备份输出的日志文档
# 调用登录获取课程等基础配置
from src.baseinfo.base_info import BaseInfo, BaseResult
from src.baseinfo.websocket_app import PhoenixWebSocketApp
from src.fileconfig.json_file_config import HandleJson

# 调用学科组件模块
from src.modules.assembly.px_en import PxEn
from src.modules.assembly.px_math import PxMath


class InitData:
    def __init__(self):
        # logger日志初始化
        self.logger = ConfigLogging().write_logging()
        # 获取钉钉
        self.ding = ConfigDing()
        # 获取当前PC的IP地址
        self.mac_info = HardwareConfig()
        # 获取学生端版本信息
        self.px_version = GetVersion()
        time.sleep(3)
        # 获取服务端学生信息【0.3.3】
        self.info_data = BaseInfo()
        # 获取课件信息
        self.status_run = BaseResult()
        # 清除课件相关数据信息
        self.basefile = BaseFile()


class PxPtAuto(InitData):
    def __init__(self, pipe):
        InitData.__init__(self)
        # 获取基础数据
        self.base_data = ReadBaseConfig()
        self.json_info = HandleJson()

        # 获取version.json中课件版本数据   editby wyb
        self.getversion = ComVersion()

        # 获取登录基础信息
        if "off" in self.base_data.get_login("en_username"):
            ipad_login = self.info_data.offline_ipad_login()
        else:
            ipad_login = self.info_data.ipad_login()
        # 获取ipad端登录数据
        self.login_token = ipad_login["token"]
        self.login_userid = ipad_login["userid"]
        self.classid = ipad_login["classID"]
        self.lessonnum = ipad_login["lessonNum"]
        self.eid = ipad_login["eid"]
        self.content = ipad_login["content"]
        self.username = ipad_login["username"]
        # 调用英语/数学组件
        self.pxen = PxEn()
        self.pxma = PxMath()
        # 消息队列
        self.pipe = pipe

    # 创建长websocket连接【0.3.3】
    def get_websocket_linking(self):
        websocket_url = self.base_data.get_http("base_new_websocket_url")
        random_string = "".join(
            random.sample(string.ascii_letters + string.digits, 32)
        )
        if self.login_token and self.login_userid:
            ws_url = (
                websocket_url
                + self.login_token
                + "&client=20&postId="
                + random_string
                + "&userId="
                + str(self.login_userid)
            )
            myapp = PhoenixWebSocketApp(ws_url, self.pipe[0])
            myapp.websocket_start()
        else:
            self.logger.error("websocket_token or userid error")

    # 组件互动【0.4】
    def handle_info(self):
        # 同步QA中心测试状态
        self.status_run.test_status(3, "")
        # PC学生端登录
        time.sleep(3)
        lt = self.info_data.pc_login(self.login_token)
        if not lt:
            self.logger.error("扫码异常")
        count = 0  # 绑定头像标记数据
        courseware_count = 0  # 获取失败课件加载次数
        courseware_camera_count = 0  # 摄像头检查
        answer_machine_count = 0  # 答题器检查
        courseware_network_count = 0  # 网络检查
        headpictures_count = 0  # 获取头像
        head_count = 0  # 获取头像失败统计次数
        showpicture = 0  # showpicture看图展示数据
        component_ids = []  # 组件ID列表
        component_names = []  # 组件名称列表
        component_name_count = None  # 组件统计数量
        call_count = 0  # 多人问答
        classname = self.status_run.get_classname()  # 动态获取课件名称
        offlesson_count = 0  # 下课统计
        stu_lists = None
        gid_lists = None
        start_time = time.time()
        # 测试结果回传数据
        content = {
            "title": r"auto-课件自动化测试报告",
            "hostname": str(self.mac_info.get_hostname()),
            "mac": str(self.mac_info.get_mac("以太网")),
            "px_version": self.px_version.get_cversion("version"),
            "username": self.username,
            "classid": classname,
            "classname": classname,
            "cid": "",
            "error_code": "",
            "error_detail": "",
            "errorReason": "",
            "componentcount": component_name_count,
            "used_time": "",
            "checkstarttime": int(start_time),
            "checkendtime": 0,
            "remote_id": self.base_data.get_login("Sunflower_name"),
            "remote_pwd": self.base_data.get_login("Sunflower_pwd"),
        }
        # magic_hot_count = 0
        while True:
            msg = self.pipe[1].recv()
            self.logger.info("ipad端接收消息：" + str(msg))
            if msg["command"] == "COURSE_READYTOCHOSE" and count == 0:
                # iPad端初始化
                time.sleep(1)
                self.info_data.ipad_init(
                    self.login_token,
                    self.classid,
                    self.lessonnum,
                    self.login_userid,
                )
                # iPad端加载上课信息同步学生端
                time.sleep(3)
                self.info_data.ipad_clickclass(
                    self.login_token,
                    self.classid,
                    self.lessonnum,
                    self.eid,
                    self.content,
                )
            # elif msg['command'] == "DEVICE_STATUS" and count == 0:
            #     device = msg['content']['device']
            #     if device["answerMachine"]["code"] != 3000 and answer_machine_count <=3:
            #         answer_machine_count +=1
            #         self.logger.warning("答题器ERROR:" + str(device))
            #         self.ding.ding_test(3, ("答题器ERROR:" + str(device)))
            #     if device["camera"]["code"] != 4000 and courseware_camera_count <=3:
            #         courseware_camera_count += 1
            #         self.logger.warning("摄像头ERROR:" + str(device))
            #         self.ding.ding_test(3, ("摄像头ERROR:" + str(device)))
            #     if device["network"]["code"] != 5000 and courseware_network_count <=3:
            #         courseware_network_count +=1
            #         self.logger.warning("网络ERROR:" + str(device))
            #         self.ding.ding_test(3, ("网络ERROR:" + str(device)))
            # 新增AI模型检测阶段
            elif msg["command"] == "SOFTWARE_STATUS" and count == 0:
                ai_code = msg["content"]["software"]["code"]
                ai_status = msg["content"]["software"]["status"]
                if ai_code == 8000 and ai_status == "fail":
                    print("AI模型检测失败")
                    self.logger.error(
                        "***************************AI模型检测失败****************************"
                    )
                    content["result_code"] = 1
                    content["result_name"] = (
                        r"【AI模型检测失败，进入待测试状态】"
                    )
                    content["error_code"] = 102
                    content["error_detail"] = str(
                        "AI模型检测失败，进入待测试状态，code码:" + str(ai_code)
                    )
                    content["errorReason"] = str(
                        "AI模型检测失败，进入待测试状态,code码:" + str(ai_code)
                    )
                    self.ding.ding_test(2, content)  # 钉钉推送 state2
                    self.status_run.test_status(
                        1, content
                    )  # 同步QA中心状态成为待测试状态
                    self.basefile.all_data()  # 清空所有课件相关数据
                    # 检测失败就退出跑课进程
                    break
                else:
                    print("AI模型检测成功")
                    self.logger.info(
                        "***************************AI模型检测成功****************************"
                    )
            elif msg["command"] == "DEVICE_STATUS" and count == 0:
                device = msg["content"]["device"]
                if (
                    device["answerMachine"]["code"] != 3000
                    and answer_machine_count <= 3
                ):
                    if answer_machine_count == 3:
                        self.logger.warning("答题器ERROR:" + str(device))
                        self.ding.ding_test(3, ("答题器ERROR:" + str(device)))
                    else:
                        answer_machine_count += 1
                if (
                    device["camera"]["code"] != 4000
                    and courseware_camera_count <= 3
                ):
                    if courseware_camera_count == 3:
                        self.logger.warning("摄像头ERROR:" + str(device))
                        self.ding.ding_test(3, ("摄像头ERROR:" + str(device)))
                    else:
                        courseware_camera_count += 1
                if (
                    device["network"]["code"] != 5000
                    and courseware_network_count <= 3
                ):
                    if courseware_network_count == 3:
                        self.logger.warning("网络ERROR:" + str(device))
                        self.ding.ding_test(3, ("网络ERROR:" + str(device)))
                    else:
                        courseware_network_count += 1
            elif msg["command"] == "COURSE_STATUS" and count == 0:
                course = msg["content"]["course"]
                if course["code"] == 9000 and courseware_count <= 3:
                    pass
                elif course["code"] != 9000 and courseware_count <= 3:
                    self.logger.warning("课件ERROR:" + str(course))
                    time.sleep(1)
                    self.info_data.get_checkaction_url(
                        self.login_token, self.classid, self.lessonnum
                    )
                    courseware_count += 1

                else:
                    self.logger.error(
                        "加载3次课件失败or课件不完整,错误ERROR:" + str(course)
                    )
                    content["result_code"] = 1
                    content["result_name"] = r"【测试不通过】"
                    content["error_code"] = 102
                    content["error_detail"] = str(
                        "加载3次课件失败or课件不完整,错误ERROR:" + str(course)
                    )
                    content["errorReason"] = str(
                        "加载3次课件失败or课件不完整,错误ERROR:" + str(course)
                    )
                    self.status_run.test_result(content, 2)
                    break
            elif msg["command"] == "GAME_INIT_READY" and count == 0:
                # ipad端获取头像
                headpictures = self.info_data.get_stu_headpictures_api(
                    self.login_token, self.classid, self.lessonnum
                )
                if not headpictures:
                    # ipad端获取头像
                    self.info_data.get_stu_headpictures_api(
                        self.login_token, self.classid, self.lessonnum
                    )
                    headpictures_count += 1
                    self.logger.warning(
                        "ipad端发送获取头像异常：" + str(headpictures_count)
                    )
                # ipad端获取未绑定学生列表
                time.sleep(1)
                stu_lists = self.info_data.get_student_sign_list_api(
                    self.login_token, self.classid, self.lessonnum, 1
                )
                # ipad端获取学生头像gid
                time.sleep(2)
                gid_lists = self.info_data.get_student_positions_api(
                    self.login_token, self.classid, self.lessonnum
                )
            elif (
                msg["command"] == "Back_Portrait_From_Screen"
                and msg["content"]["code"] == 1002
            ):
                head_count += 1
                self.logger.warning("识别端未接收到可用视频流,返回1002,ERROR")
                if head_count >= 3:
                    self.logger.error("识别端超过三次获取视频异常1002")
                    break
            elif msg["command"] == "GET_AVATARS_BACK" and count == 0:
                if not gid_lists:
                    self.logger.error("ipad端第一次获取头像失败")
                    gid_lists = self.info_data.get_student_positions_api(
                        self.login_token, self.classid, self.lessonnum
                    )
                if msg["content"]["code"] == 1000 and gid_lists and stu_lists:
                    time.sleep(1)
                    # ipad端发送绑定头像
                    bindstu = self.info_data.send_bind_headpictures(
                        gid_lists,
                        stu_lists,
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                    )
                    if bindstu:
                        count += 1
                    else:
                        self.info_data.send_bind_headpictures(
                            gid_lists,
                            stu_lists,
                            self.login_token,
                            self.classid,
                            self.lessonnum,
                        )
                        count += 1
                else:
                    self.logger.error(
                        "ipad端绑定头像ERROR："
                        + str(msg)
                        + str(gid_lists)
                        + str(stu_lists)
                    )
                    content["result_code"] = 1
                    content["result_name"] = r"【测试不通过】"
                    content["error_code"] = 103
                    content["error_detail"] = str(
                        "ipad端绑定头像ERROR："
                        + str(msg)
                        + str(gid_lists)
                        + str(stu_lists)
                    )
                    content["errorReason"] = str(
                        "ipad端绑定头像ERROR："
                        + str(msg)
                        + str(gid_lists)
                        + str(stu_lists)
                    )
                    self.status_run.test_result(content, 2)
                    break
            else:
                # ipad绑定头像状态
                if msg["command"] == "BIND_AVATARS_BACK":
                    if msg["content"]["code"] == 1000:
                        # ipad端点击上课
                        time.sleep(1)
                        self.info_data.start_lesson_api(
                            self.login_token, self.classid, self.lessonnum
                        )
                        time.sleep(1)
                        # 播放视频
                        self.info_data.get_video_play_stu_api(
                            self.login_token,
                            self.classid,
                            self.lessonnum,
                            self.login_userid,
                            self.lessonnum,
                        )
                    elif msg["content"]["code"] == 1104:
                        time.sleep(1)
                        self.logger.error("绑定头像已失效ERROR")
                        count -= 1
                        result_data = self.info_data.get_stu_headpictures_api(
                            self.login_token, self.classid, self.lessonnum
                        )
                        if result_data:
                            time.sleep(2)
                            # 获取未绑定的学生列表
                            stu_lists = (
                                self.info_data.get_student_sign_list_api(
                                    self.login_token,
                                    self.classid,
                                    self.lessonnum,
                                    1,
                                )
                            )
                            # 获取识别到的学生头像gid
                            gid_lists = (
                                self.info_data.get_student_positions_api(
                                    self.login_token,
                                    self.classid,
                                    self.lessonnum,
                                )
                            )
                # 倒计时组件
                elif msg["command"] == "START_COUNTDOWN":
                    time.sleep(3)
                    self.pxen.handle_countdown(
                        self.login_token, self.classid, self.lessonnum
                    )
                # 1V1点人和选人问答组件
                elif msg["command"] in [
                    "RECOGNIZE_HANDSUP_END",
                    "RECOGNIZE_HANDSUP_END_PRO",
                ]:
                    signl_stu_lists = self.info_data.get_student_sign_list_api(
                        self.login_token, self.classid, self.lessonnum, 2
                    )
                    self.pxen.en_handle_onetoone(
                        msg,
                        signl_stu_lists,
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        msg["command"],
                    )
                # 看图展示组件
                elif msg["command"] == "SELECTTALKING_START":
                    global stulist_sp
                    stulist_sp = self.pxen.get_student_picture_list_api(
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        msg["componentID"],
                    )  # ipad获取展示学生信息列表
                    self.pxen.get_request_selecttalking_info_api(
                        self.login_token, self.classid, self.lessonnum
                    )  # iPad端发送获取学生端信息
                elif (
                    msg["command"] == "SYNC_CLASS_BTN_STATUS"
                    and msg["content"]["breakPoint"]["status"] == 1
                ):
                    self.logger.error(
                        "*************此处有断点续播*******************"
                    )

                elif msg["command"] == "BACK_SELECTTALKING_INFO":
                    global picture_urls
                    picture_urls = msg["content"][
                        "defaultUrls"
                    ]  # 获取看图展示所有默认图片
                    self.pxen.handle_showpicture_sendimage(
                        stulist_sp,
                        picture_urls,
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                    )  # 发送看图展示表杨点赞
                elif (
                    msg["command"] == "BACK_SELECTTALKING_STU"
                ):  # 发送默认图后学生端返回展示中的学生信息
                    if self.pxen.handle_showpicture_praise(
                        self.login_token, self.classid, self.lessonnum
                    ):
                        showpicture += 1
                    time.sleep(1)
                elif msg["command"] == "SElECTTALKING_PRAISE_END":
                    if showpicture < 2:
                        self.pxen.handle_showpicture_sendimage(
                            stulist_sp,
                            picture_urls,
                            self.login_token,
                            self.classid,
                            self.lessonnum,
                        )  # 发送看图展示表杨点赞
                    else:
                        self.pxen.handle_showpicture_end(
                            self.login_token, self.classid, self.lessonnum
                        )
                        time.sleep(1)
                # RolePlay组件
                elif msg["command"] == "READ_START":
                    sentence_order = msg["content"]["sentenceOrder"]
                    if sentence_order != "":
                        time.sleep(2)  # 添加2秒钟等待时间，防止过早操作卡住
                        self.pxen.role_paly_repeat(
                            self.login_token,
                            self.classid,
                            self.lessonnum,
                            sentence_order,
                        )
                    else:
                        self.logger.error("sentenceOrder异常:" + str(msg))
                # 统计组件数量【AIB】
                elif msg["command"] == "COMPONENT_END":
                    component_ids.append(
                        msg["content"]["component"]["componentID"]
                    )
                    # 增加课件名称加版本号到component_names中   v1.2                     editby wyb
                    versioned_component_name = self.getversion.get_version_name(
                        msg["content"]["component"]["componentName"]
                    )
                    component_names.append(versioned_component_name)
                    # component_names.append(msg['content']['component']['componentName'])
                    self.logger.info("统计组件ID：" + str(component_ids))
                    self.logger.info("统计组件NAME：" + str(component_names))
                    # 发放积分【AIB】
                    if component_ids and len(component_ids) % 3 == 0:
                        try:
                            self.info_data.point_change(
                                self.login_token, self.classid, self.lessonnum
                            )
                            # remaining_tutor_points = self.info_data.get_lesson_points_api(self.login_token, self.classid,
                            #                                                      self.lessonnum)
                            # isbind_stuids = self.info_data.get_student_sign_list_api(self.login_token, self.classid,
                            #                                                            self.lessonnum, 2)
                            # self.info_data.change_points_api(self.login_token, self.classid, self.lessonnum,
                            #                                isbind_stuids, remaining_tutor_points)
                        except Exception as e:
                            self.logger.error("发放积分异常：" + str(e))
                            continue
                    # 发放勋章/课后大表扬
                    if len(component_ids) == 1:
                        try:
                            student_id_list = (
                                self.pxma.get_student_goal_by_type_id(
                                    self.login_token,
                                    self.classid,
                                    self.lessonnum,
                                    1,
                                )
                            )
                            self.logger.info(
                                "课后大表扬获取学生列表："
                                + str(student_id_list)
                            )
                            self.pxma.set_student_medals(
                                self.login_token,
                                self.classid,
                                self.lessonnum,
                                student_id_list,
                            )
                        except Exception as e:
                            self.logger.error("发放勋章/课后大表扬：", e)
                            continue
                # 魔法帽组件
                elif msg["command"] == "MAGIC_HAT_START":
                    global component_id, componentName, stuNameone, stuIDone
                    component_id = msg["content"]["componentID"]
                    componentName = msg["content"]["componentName"]
                    stuNameone = msg["content"]["nowStudent"]["stuName"]
                    stuIDone = msg["content"]["nowStudent"]["stuID"]
                elif msg["command"] in [
                    "MAGIC_HAT_NEXT_SUCCESS",
                    "MAGIC_HAT_WANT_NEXT",
                    "MAGIC_HAT_END",
                ]:
                    self.pxen.magic_hot(
                        msg,
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        component_id,
                        componentName,
                    )
                # 数学判题【做题投屏】
                elif msg["command"] == "GROUP_QUESTION_ANSWER_START":
                    self.pxma.question_answer(
                        self.login_token, self.classid, self.lessonnum
                    )
                # 多人问答【数学】
                elif msg["command"] == "CALLTHEROLL_START":
                    callcount = self.pxma.many_ask(
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        call_count,
                        msg,
                    )
                    call_count = callcount
                # 快速点名【数学】
                elif msg["command"] == "QUICK_CALLTHEROLL_START":
                    ma_stuid = msg["content"]["nowStudent"]["stuID"]
                    ma_stuname = msg["content"]["nowStudent"]["name"]
                    math_component_id = msg["componentID"]
                    math_component_type = msg["content"]["componentName"]
                    ma_strategy = "math_scan-quick-qa"
                    ma_stu_lists = self.info_data.get_tagstu_list(
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        math_component_id,
                        math_component_type,
                        ma_strategy,
                    )
                    self.pxma.quick_qa(
                        self.login_token,
                        self.classid,
                        self.lessonnum,
                        math_component_type,
                        ma_stuid,
                        ma_stuname,
                        ma_stu_lists,
                    )
                # 彩蛋【数学】
                elif msg["command"] == "EGG_CONFIRM":
                    self.pxma.egg_play(
                        self.login_token, self.classid, self.lessonnum
                    )
                # ipad端下课【AIB】
                elif msg["command"] in ["LESSON_END", "OFF_LESSON"]:
                    end_time = time.time()
                    content["used_time"] = str(int(end_time - start_time)) + "S"
                    content["checkendtime"] = int(end_time)
                    component_name_count = self.json_info.count_list(
                        component_names
                    )  # 课程组件统计
                    content["componentcount"] = component_name_count
                    match_cid = self.json_info.match_cid(
                        component_ids
                    )  # 匹配last_cid一致性
                    if len(component_ids):
                        content["cid"] = component_ids[-1]
                    self.logger.info(
                        "课程组件统计：" + str(component_name_count)
                    )
                    if msg["command"] == "LESSON_END":
                        offlesson_status = None
                        while offlesson_count < 3 and not offlesson_status:
                            offlesson_status = self.info_data.end_lesson_api(
                                self.login_token, self.classid, self.lessonnum
                            )
                            time.sleep(1)
                            offlesson_count += 1
                        if offlesson_count >= 3:
                            content["result_code"] = 1
                            content["result_name"] = r"【测试不通过】"
                            content["error_code"] = 103  # 103课中异常
                            content["error_detail"] = "LESSON_END_ERROR"
                            content["errorReason"] = "LESSON_END_ERROR"
                            # content['report'] = r'【测试不通过】'
                            # content['error_detail'] = "LESSON_END_ERROR"
                            self.logger.info(content)  # 统计记录日志
                            self.status_run.test_result(content, 2)
                            break
                    elif msg["command"] == "OFF_LESSON":
                        self.info_data.get_video_end_stu_api(
                            self.login_token,
                            self.classid,
                            self.lessonnum,
                            self.login_userid,
                        )
                        if match_cid:
                            content["result_code"] = 0
                            content["result_name"] = r"【测试通过】"
                            content["error_code"] = 100  # 100测试通过
                            content["error_detail"] = "无"
                            content["errorReason"] = "无"
                            # content['report'] = r'【测试通过】'
                            # content['error_detail'] = "无"
                            self.logger.info(content)  # 统计记录日志
                            self.status_run.test_result(content, 1)
                            break
                        else:
                            content["result_code"] = 1
                            content["result_name"] = r"【测试不通过】"
                            content["error_code"] = 104  # 104课后异常
                            content["error_detail"] = "last_cid not match"
                            content["errorReason"] = "last_cid not match"
                            # content['report'] = r'【测试不通过】'
                            # content['error_detail'] = "last_cid not match"
                            self.logger.info(content)  # 统计记录日志
                            self.status_run.test_result(content, 2)
                            break


if __name__ == "__main__":
    from src.config.platform_compat import get_platform, is_windows

    platform = get_platform()
    print(f"当前操作系统: {platform.system}")

    # 确保必要的目录存在
    from src.settings import ensure_directories

    ensure_directories()

    basefile = BaseFile()
    basefile.update_pid("1")
    # 备份日志
    logger = ConfigLogging()
    logger.handle_log()
    # 清除旧的截图文件
    screen = ConfigScreen()
    screen.del_screen_image()
    # 校验mock答题器文件
    kid = PidConfig()
    kid.copy_mockfile()

    # 上课前清理端无用进程（跨平台）
    if is_windows():
        kill_cmd = r"python /px_pt_auto/BaseConfig/kill_process.py"
    else:
        # macOS/Linux使用相对路径或配置路径
        kill_cmd = f"python3 {os.path.join(platform.get_user_home(), 'px_pt_auto', 'BaseConfig', 'kill_process.py')}"

    os.system(kill_cmd)
    time.sleep(2)

    # 启动学生端（仅Windows支持）
    if is_windows():
        start_cmd = r"/px_pt_auto/BaseConfig/start_phoenix.bat"
        os.system(start_cmd)
    else:
        print("警告: Phoenix学生端仅支持Windows系统，跳过启动")
        print("在macOS/Linux上，部分功能可能不可用")

    # 消息队列
    pipe = Pipe()

    # Windows下多进程必须加此代码
    if is_windows():
        multiprocessing.freeze_support()

    # 启动虚拟摄像头（仅Windows支持）
    kid.run_kill()
    if is_windows():
        pm = Process(target=run_vcamtest, args=("e",))  # 必须加,号
        pm.start()
        time.sleep(1)
    else:
        print("虚拟摄像头功能仅支持Windows系统，跳过")
        pm = None

    # 截图
    psc = Process(target=screen.run_screen)
    psc.start()

    # 跑三个小时超时策略
    three = Process(target=main)
    three.start()

    # 实例化主进程
    pea = PxPtAuto(pipe)

    # iPad端websocket
    pw = Process(target=pea.get_websocket_linking)
    pw.start()

    # 课中组件交互主程序
    pea.handle_info()

    # 清理进程
    pw.terminate()
    if pm:
        pm.terminate()
    psc.terminate()
    # 结束超过三小时策略
    three.terminate()
    # 清理视频播放进程
    kid.run_kill()
    os.system(r"python /px_pt_auto/BaseConfig/kill_process.py")
