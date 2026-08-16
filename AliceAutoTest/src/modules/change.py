# !/usr/bin/python3
import hashlib
import json
import os
import random
import re
import socket
import time
from configparser import ConfigParser

import requests

from src import settings


# 获取课件的信息
def get_course_info():
    with open(settings.COURSEWARE_DATA) as f:
        lines = f.read().splitlines()
        courseid = lines[1]
        version = lines[2]
        client_version = lines[3]
        subject = lines[4]
    return courseid, version, client_version, subject


# 改变一次课件的结果
def change_one_status(status, start_time, errors):
    myname = socket.getfqdn(socket.gethostname())
    courseid, version, client_version, subject = get_course_info()
    statu_parm = {
        "coursewareId": courseid,
        "coursewareVersion": int(version),
        "checkBeginDate": int(start_time * 1000),
        "checkEndDate": int(time.time() * 1000),
        "checkStatus": int(status),
        "clientVersion": get_client_info().get("clientVersion"),
        "macAddress": "10.33.14.46",
        "computerName": myname,
        "errorReason": errors,
        "testAccount": get_client_info().get("en_username"),
        "lastVideoId": get_client_info().get("lastVideoId"),
        "componentStatistics": get_client_info().get("componentStatistics"),
    }
    postid = random.randint(0, 1000)
    headers = {
        "Content-Type": "application/json",
        "T-px-Post-ID": str(postid),
        "T-px-Trace-ID": "1",
        "T-px-Validate-Token": "ss",
    }
    requests.post(
        settings.CHANGE_ONE_STATUS_URL,
        headers=headers,
        data=json.dumps(statu_parm),
    )


# 改变最终状态
def change_status(status, errors):
    # 获取本机电脑名
    myname = socket.getfqdn(socket.gethostname())
    courseid, version, client_version, subject = get_course_info()
    statu_parm = {
        "coursewareId": courseid,
        "coursewareVersion": int(version),
        "checkStatus": int(status),
        "differenceResult": "www.baidu.com",
        "checkResult": {
            "errorReason": errors,
            "clientVersion": client_version,
            "computerName": myname,
            "macAddress": "",
            "testAccount": "",
            "lastVideoId": "",
            "componentStatistics": "",
        },
    }
    postid = random.randint(0, 1000)
    headers = {
        "Content-Type": "application/json",
        "T-px-Post-ID": str(postid),
        "T-px-Trace-ID": "1",
        "T-px-Validate-Token": "ss",
    }
    requests.post(
        settings.CHANGE_STATUS_URL, headers=headers, data=json.dumps(statu_parm)
    )


# 获取跑课端的账号以及版本状态
def get_client_info():
    target = ConfigParser()
    target.read(settings.NUM_PATH, encoding="utf-8")
    dir_path = dict(target.items("LOGIN"))  # 获取到配置文件中所有关于登录的信息
    target1 = ConfigParser()
    target1.read(settings.STATUS_PATH, encoding="utf-8")
    status_dic = dict(target1.items("status"))  # 获取到配置文件中关于端的信息
    status_dic.setdefault(
        "en_username", dir_path.get("en_username")
    )  # 将端的账号和端的状态合并在一起
    print(status_dic)
    return status_dic


# 自动跑课发送钉钉消息
def send_ding(error):
    # 获取本机电脑名
    myname = socket.getfqdn(socket.gethostname())
    # 获取本机ip
    myaddr = socket.gethostbyname(myname)
    # 获取本机向日葵账号
    target = ConfigParser()
    target.read(settings.STATUS_PATH, encoding="utf-8")
    xrk_info = dict(target.items("XRK"))
    # 课件json应有内容资源项
    # remotefiles = sys.argv[1]
    # # 课件实际下载内容资源项
    # localfiles = sys.argv[2]

    # 读取到当前课件的信息
    with open(settings.COURSEWARE_DATA) as f:
        lines = f.read().splitlines()
        courseid = lines[1]
        version = lines[2]
        num = lines[-1]
    con = {
        "msgtype": "markdown",
        "markdown": {
            "title": "自动化跑课测试报告",
            "text": "#### aib测试报告\n"
            + f"> 本机名 : {myname}\n\n"
            + f"> 本机ip : {myaddr}\n\n"
            + f"> 本机向日葵 : {xrk_info.get('sunflower')}:\n\n "
            + "课件id："
            + courseid
            + "\n\n"
            + "课件版本： "
            + version
            + "\n\n"
            + "总跑课次数:  "
            + num
            + "\n\n"
            + "错误信息:"
            + "\n\n"
            + "- "
            + error
            + "\n\n",
            # "***  \n" +
            # f"- 课件json应有内容资源项: {remotefiles} \n" +
            # f"- 课件实际下载内容资源项: {localfiles}  \n"
        },
        "at": {"atMobiles": [], "isAtAll": False},
    }
    requests.post(url=settings.ERROR_COURSE, json=con)
    return myaddr, xrk_info


# 寻找第一个待测试的课件
def find_first_download_course(courses_list, exclude_aib=True):
    """
    寻找第一个待测试课件
    :param courses_list: 待测试课件列表
    :param exclude_aib: True为非AIB，False为AIB
    :return: 没找到，返回None
    """
    for i in courses_list:
        if exclude_aib:
            rule = settings.NO_AIB
            print("我不是aib课件")
        else:
            rule = settings.AIB_RULE
            print("我是aib课件")
        patten = re.compile(rule)
        res = patten.match(i["clientVersion"])
        if res:
            return i
    return None


# 调用url
def fun_request(url, postid, parm=None):
    # url = "http://pxtest.facethink.com/courseCheck/action/autoCheck/getPendCheckCoursewares"
    headers = {
        "Content-Type": "application/json",
        "T-px-Post-ID": postid,
        "T-px-Trace-ID": "1",
        "T-px-Validate-Token": "ss",
    }
    response = requests.post(url, headers=headers, data=parm)
    # data = response.content
    # print(response)
    data = json.loads(response.content)
    return data


#
#
# # 创建文件
def write_file(file_path):
    with open(file_path, "w") as courseware_data:
        courseware_data.write("1")


# #
# # 获取下载文件的md5值
def get_md5(file_path):
    """计算MD5"""
    with open(file_path, "rb") as input_file:
        return hashlib.md5(input_file.read()).hexdigest()


# #
# #
# # 获取本地文件
def get_filenames(file_path):
    files_list = []
    for root, _dirs, files in os.walk(file_path):
        for file in files:
            local_file = os.path.join(root, file)
            a = local_file.replace(file_path, "")
            files_list.append(a[1:])
    files_list = [i.replace("\\", "/") for i in files_list]
    return files_list


def write_result(pass_or_fail, info):
    path = time.strftime("%Y_%m_%d", time.localtime(time.time())) + ".json"
    path = os.path.join(settings.AUTO_OUTPUT, path)
    if os.path.exists(path):
        with open(path, "r+", encoding="utf-8") as f:
            item = json.loads(f.read())
            item["total"] += 1
            item[pass_or_fail] += 1
            courseid, version, client_version, subject = get_course_info()
            item[pass_or_fail + "_list"].append(
                {
                    "courseID": courseid,
                    "version": version,
                    "subject": subject,
                    "info": info,
                }
            )
            f.seek(0)
            f.write(json.dumps(item, indent=2, ensure_ascii=False))
    else:
        with open(path, "w", encoding="utf-8") as f:
            item = dict()
            item["total"] = 1
            if pass_or_fail == "pass":
                item[pass_or_fail] = 1
                item["fail"] = 0
                item["fail_list"] = []
            else:
                item[pass_or_fail] = 1
                item["pass"] = 0
                item["pass_list"] = []
            courseid, version, client_version, subject = get_course_info()
            item[pass_or_fail + "_list"] = [
                {
                    "courseID": courseid,
                    "version": version,
                    "subject": subject,
                    "info": info,
                }
            ]
            f.write(json.dumps(item, indent=2, ensure_ascii=False))


a = find_first_download_course(
    [
        {
            "Content-Type": "application/json",
            "T-px-Post-ID": "123",
            "T-px-Trace-ID": "1",
            "T-px-Validate-Token": "ss",
            "clientVersion": "2.1.1.0",
        }
    ],
    exclude_aib=False,
)
print(a)
