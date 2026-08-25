import configparser
import getpass
import os
import re
from pathlib import Path

from src.config.platform_compat import get_platform, is_windows

_MISSING = object()


class ConfigParser:
    def __init__(self):
        # 读取config.ini基础路径内容
        self.cf = configparser.ConfigParser()

    # 设置config.ini格式
    def set_format(self, config_path):
        try:
            self.cf.read(config_path, encoding="UTF-8")
        except Exception as e:
            print(e)
            content = Path(config_path).read_text(encoding="utf-8")
            # Window下用记事本打开配置文件并修改保存后，编码为UNICODE或UTF-8的文件的文件头
            # 会被相应的加上\xff\xfe（\xff\xfe）或\xef\xbb\xbf或\ufeff，然后再传递给ConfigParser解析的时候会出错
            # ，因此解析之前，先替换掉
            content = re.sub(r"\ufeff", "", content)
            Path(config_path).write_text(content, encoding="utf-8")
            self.cf.read(config_path, encoding="UTF-8")


# 读取config.ini配置文件的配置数据
class ReadBaseConfig(ConfigParser):
    def __init__(self):
        ConfigParser.__init__(self)
        project_root = Path(__file__).resolve().parents[2]
        configured_path = os.getenv("ALICE_AUTOTEST_CONFIG")
        possible_paths = [
            configured_path,
            project_root / "config" / "config.local.ini",
            project_root / "config" / "config.ini",
            Path.cwd() / "config" / "config.local.ini",
            Path.cwd() / "config" / "config.ini",
            project_root / "config" / "config.example.ini",
        ]

        self.config_path = None
        for path in possible_paths:
            if path and Path(path).is_file():
                self.config_path = str(path)
                break

        if not self.config_path:
            raise FileNotFoundError(
                "未找到 AliceAutoTest 配置。请复制 config/config.example.ini "
                "为 config/config.local.ini，或设置 ALICE_AUTOTEST_CONFIG。"
            )

        self.set_format(self.config_path)

    def _get(self, section, name, fallback=_MISSING):
        env_name = f"ALICE_AUTOTEST_{section}_{name}".upper()
        env_value = os.getenv(env_name)
        if env_value is not None:
            return env_value
        if fallback is _MISSING:
            return self.cf.get(section, name)
        return self.cf.get(section, name, fallback=fallback)

    # 获取数据库基础配置信息
    def get_db(self, name):
        return self._get("DB", name)

    # 获取发送邮件的基础数据信息
    def get_smtp(self, name):
        return self._get("SMTP", name)

    def get_casepath(self, name):
        return self._get("CASE_PATH", name)

    # 获取登录的账号信息
    def get_login(self, name):
        return self._get("LOGIN", name)

    # 获取接口的URL和参数配置信息
    def get_http(self, name):
        return self._get("HTTP", name)

    # 获取存储报告的基础路径
    def get_reportpath(self, name):
        return self._get("REPORT_PATH", name)

    # 获取参数化文档路径
    def get_filepath(self, name):
        return self._get("FILE_PATH", name)

    def get_dingtalk(self, name):
        """读取通知配置；敏感值应放在环境变量或本地配置中。"""
        return self._get("DINGTALK", name, fallback="")

    def get_contacts(self, name):
        """读取联系人配置；未配置时返回空字符串。"""
        return self._get("CONTACTS", name, fallback="")


class GetLogin(ReadBaseConfig):
    def getlogin(self, subject):  # subject 1数学 2英语、
        result_data = {
            "username": self.get_login("ma_username"),
            "password": self.get_login("ma_password"),
            "client": "20",
            "clientVersion": "",
        }
        if int(self.get_login("username_status")) == 1:
            pass
        elif int(self.get_login("username_status")) == 2:
            result_data["username"] = self.get_login("en_username")
            result_data["password"] = self.get_login("en_password")
        else:
            if subject:
                if int(subject) == 1:
                    pass
                elif int(subject) == 2:
                    result_data["username"] = self.get_login("en_username")
                    result_data["password"] = self.get_login("en_password")
                else:
                    pass
            else:
                pass
        return result_data


class GetVersion(ReadBaseConfig):
    def __init__(self):
        ReadBaseConfig.__init__(self)
        self._platform = get_platform()

    # 获取PC端版本
    def get_cversion(self, name):
        if not is_windows():
            return "unknown"
        try:
            configname = self.get_filepath("config_name")
            config_path = os.path.abspath(
                os.path.join(rf"C:\Users\{getpass.getuser()}", configname)
            )  # 返回规范化的绝对路径
            self.set_format(config_path)
            self.cf.read(
                config_path, encoding="utf-8"
            )  # 获取配置文件数据内容，注意字符编码格式
            value = self.cf.get("Server", name)
            return value
        except Exception as e:
            print(f"获取版本信息失败: {e}")
            return "unknown"

    # 获取答题器mock
    def get_mock(self):
        mockname = self.get_filepath("mock_pxanswer")
        if is_windows():
            mockpath = os.path.abspath(
                os.path.join(rf"C:\Users\{getpass.getuser()}", mockname)
            )
        else:
            # macOS/Linux: 使用应用数据目录
            mockpath = os.path.join(
                self._platform.get_app_data_path(), mockname
            )
        return mockpath


class ListConfig:
    def __init__(self):
        pass

    def countlist(self, listname):
        result_list = {}
        for i in set(listname):
            result_list.update({i: listname.count(i)})
        return result_list


if __name__ == "__main__":
    data = GetVersion()
    print(data.get_mock())
    print(data.get_cversion("version"))
