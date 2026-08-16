# -*- conding:utf-8 -*-
# @Time : 2019/3/31 16:25
# @Author : liuqi
# @File : base_websocket.py
# @Software: PyCharm
import json
import os
import random
import string
import time

import websocket
from websocket import WebSocketApp

try:
    import thread
except ImportError:
    import _thread as thread


class PhoenixWebSocketApp:
    ws_data = {
        "from": "20",  # 20表示辅导端
        "to": "0",  # 接收人0表示服务端
        "command": "RTI_HEART",  # "RTI_HEART"心跳连接命令
        "content": {},
        "requireHBCheck": True,
    }
    json_data = json.dumps(ws_data)

    def __init__(self, ws_url, pipe=None):
        super().__init__()
        self.pipe = pipe
        self.url = ws_url
        self.ws = None

    def on_message(self, message):
        # print("message:", message)
        server_message = json.loads(message)
        if server_message["command"] != "RTI_HEART":
            if self.pipe:
                self.pipe.send(server_message)
            else:
                print(server_message)

    def on_error(self, error):
        print(error)

    def on_close(self):
        print("### closed ###")

    def on_open(self):
        def run(*args):
            while True:
                time.sleep(8)
                self.ws.send(self.json_data)

        thread.start_new_thread(run, ())

    def websocket_start(self):
        websocket.enableTrace(True)
        self.ws = WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        self.ws.on_open = self.on_open
        self.ws.run_forever()


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

    ws = PhoenixWebSocketApp(ws_url)
    ws.websocket_start()
