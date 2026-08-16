# -*- conding:utf-8 -*-
# @Time : 2019/3/31 16:25
# @Author : liuqi
# @File : base_websocket.py
# @Software: PyCharm
import contextlib
import json
import os
import random
import string

import websocket

with contextlib.suppress(ImportError):
    pass


# websocket长连接
class WebSocketClient:
    def __init__(self, url, send_data):
        self.url = url
        self.send_data = send_data

    # 监听server端发送的消息
    def on_message(self, ws, message):
        get_server_command = json.loads(message)
        print(get_server_command)  # 服务端返回的消息指令写入到消息队列中

    # 监听websocket连接的错误
    def on_error(self, ws, error):
        print("websocket-error:", ws, error)

    # 关闭websocket连接
    def on_close(self, ws):
        print("### closed ###", ws)

    # 创建websocket连接
    def on_open(self, ws):
        ws.send(self.send_data)

    def stop(self, ws):
        ws.close()

    # 外部调用接口
    def start(self):
        websocket.enableTrace(True)
        ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        # 创建websocket连接
        ws.on_open = self.on_open
        # 设置超时重连
        ws.run_forever(ping_interval=5, ping_timeout=3)
        # ws.run_forever()


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

    ws_data = {
        "from": "20",  # 20表示辅导端
        "to": "0",  # 接收人0表示服务端
        "command": "RTI_HEART",  # "RTI_HEART"心跳连接命令
        "content": {},
        "requireHBCheck": True,
    }
    json_data = json.dumps(ws_data)

    websocket.enableTrace(True)
    ws = WebSocketClient(ws_url, json_data)
    ws.start()
