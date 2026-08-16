#!/usr/bin/env python
# @Time    : 2020/2/19 20:02
# @Author  : liuqi
# @Site    :
# @File    : pipe_config.py
# @Software: PyCharm

import json
import os
import random
import string

import websocket

try:
    import thread
except ImportError:
    import _thread as thread
import time


class ConfigPipe:
    def __init__(self):
        pass


ws_data = {
    "from": "20",  # 20表示辅导端
    "to": "0",  # 接收人0表示服务端
    "command": "RTI_HEART",  # "RTI_HEART"心跳连接命令
    "content": {},
    "requireHBCheck": True,
}
json_data = json.dumps(ws_data)


def on_message(ws, message):
    print("message:", message)


def on_error(ws, error):
    print(error)


def on_close(ws):
    print("### closed ###")


def on_open(ws):
    def run(*args):
        while True:
            time.sleep(7)
            ws.send(json_data)

    thread.start_new_thread(run, ())


def websocket_start(ws_url):
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        ws_url, on_message=on_message, on_error=on_error, on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()


if __name__ == "__main__":
    websocket_url = os.getenv("ALICE_AUTOTEST_WEBSOCKET_URL", "")
    token = os.getenv("ALICE_AUTOTEST_WEBSOCKET_TOKEN", "")
    userid = os.getenv("ALICE_AUTOTEST_USER_ID", "")
    if not all((websocket_url, token, userid)):
        raise SystemExit("请配置 WebSocket URL、Token 和用户 ID 环境变量")
    random_string = "".join(
        random.sample(string.ascii_letters + string.digits, 32)
    )
    ws_url = (
        websocket_url
        + token
        + "&client=20&postId="
        + random_string
        + "&userId="
        + userid
    )
    websocket_start(ws_url)
