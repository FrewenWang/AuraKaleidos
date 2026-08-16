# demo.py
from chat_push import push

if __name__ == "__main__":
    msg = "【自动化测试通知】\n测试执行完成，用例全部跑完！"

    # 单独推送
    push.send_dingtalk(msg)
    push.send_feishu(msg)
    push.send_wecom(msg)

    # 一次性推所有平台
    # push.send_all(msg)
