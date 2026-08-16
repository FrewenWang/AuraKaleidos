import requests
from push_config import DINGTALK_WEBHOOK, FEISHU_WEBHOOK, WECOM_WEBHOOK

# 超时时间
TIMEOUT = 10


class ChatPush:
    @staticmethod
    def send_dingtalk(content: str) -> bool:
        """发送钉钉群消息"""
        url = DINGTALK_WEBHOOK
        payload = {"msgtype": "text", "text": {"content": content}}
        try:
            res = requests.post(url, json=payload, timeout=TIMEOUT)
            return res.json().get("errcode") == 0
        except Exception as e:
            print(f"钉钉推送异常: {e}")
            return False

    @staticmethod
    def send_feishu(content: str) -> bool:
        """发送飞书群消息"""
        url = FEISHU_WEBHOOK
        payload = {"msg_type": "text", "content": {"text": content}}
        try:
            res = requests.post(url, json=payload, timeout=TIMEOUT)
            return res.json().get("code") == 0
        except Exception as e:
            print(f"飞书推送异常: {e}")
            return False

    @staticmethod
    def send_wecom(content: str) -> bool:
        """发送企业微信群消息"""
        url = WECOM_WEBHOOK
        payload = {"msgtype": "text", "text": {"content": content}}
        try:
            res = requests.post(url, json=payload, timeout=TIMEOUT)
            return res.json().get("errcode") == 0
        except Exception as e:
            print(f"企业微信推送异常: {e}")
            return False

    @staticmethod
    def send_all(content: str) -> None:
        """一键推送到所有平台"""
        ChatPush.send_dingtalk(content)
        ChatPush.send_feishu(content)
        ChatPush.send_wecom(content)


# 对外实例（方便直接调用）
push = ChatPush()
