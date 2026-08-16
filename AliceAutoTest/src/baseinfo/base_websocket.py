# -*- conding:utf-8 -*-
# @Time : 2019/3/31 16:25
# @Author : liuqi
# @File : base_websocket.py
# @Software: PyCharm
import contextlib
import json
import time

import websocket

with contextlib.suppress(ImportError):
    pass


# websocket长连接
class WebSocketClient:
    def __init__(self, url, send_data, pipe):
        self.url = url
        self.send_data = send_data
        self.q = pipe

    # 监听server端发送的消息
    def on_message(self, ws, message):
        get_server_command = json.loads(message)
        self.q.send(get_server_command)  # 服务端返回的消息指令写入到消息队列中

    # 监听websocket连接的错误
    def on_error(self, ws, error):
        print("websocket-error:", ws, error)

    # 关闭websocket连接
    def on_close(self, ws):
        print("### closed ###", ws)

    # 创建websocket连接
    def on_open(self, ws):
        while True:
            ws.send(self.send_data)
            time.sleep(10)

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
    url = "ws://echo.websocket.org/"
    send_data = r"sajfljsdlkjflksdj"
    data = WebSocketClient(url, send_data)
    data.start()
    # data.ws.on_message()
