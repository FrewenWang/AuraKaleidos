import socket

from src.config.config_logging import ConfigLogging
from src.config.config_mysql import MySqlConfig
from src.utils.other.change_status import ChangeStatus


class RunTimes:
    def __init__(self, content):
        self.mysqlconn = MySqlConfig()
        self.logger = ConfigLogging().write_logging()
        self.content = content
        self.socket = socket
        self.change_status = ChangeStatus()
        # 延迟导入避免循环依赖
        from src.baseinfo.base_info import BaseResult

        self.status_run = BaseResult()

    # 获取测试通过的次数
    def get_pass_count(self):
        content = self.content
        classid = content["classid"]
        sql = f'SELECT COUNT(*) AS "次数" FROM courseware WHERE  courseware_id="{classid}" AND result_name="【测试通过】"'
        return self.mysqlconn.readmysql(sql)

    # 插入数据。此处暂时用不到
    def inserresult(self):
        content = self.content
        myname = self.socket.gethostname()
        myaddr = self.socket.gethostbyname(myname)
        classid = content["classid"]
        self.mysqlconn.insertmysql(
            f"INSERT INTO courseware_paoke_info VALUES({classid},'690709798',{myaddr},'测试通过')"
        )

    def test_runtime(self):
        times = self.get_pass_count()[0]["次数"]
        self.logger.info("当前课件跑课次数" + str(times))
        content = self.content
        if times >= 3:
            self.logger.info(
                "最终结果："
                + content["result_name"]
                + "已上报钉钉消息和课件中心"
            )
            print("最终结果：", content["result_name"], "已上报钉钉消息")
            self.status_run.test_status(4, content)
        else:
            self.logger.info(
                "获取当前测试结果"
                + content["result_name"]
                + "已上报钉钉消息和课件中心"
            )
            self.logger.info("当前课件信息:" + content["classid"])
            self.logger.info("跑课状态：" + "已开始重新跑课,进入待测试状态")
            print("获取当前测试结果", content["result_name"], "已上报钉钉消息")
            print("当前课件信息:", content["classid"])
            print("跑课状态：", "已开始重新跑课，进入待测试状态")
            self.change_status.fun_request(content)
            # self.status_fun.test_result(content,1)#############这里要重新跑，要加入新的方法


if __name__ == "__main__":
    content = {"classid": "ace>>3", "result_name": "【测试不通过】"}
    # print(runtime.tongjicishu(content)[0]['次数'])
    runtime = RunTimes(content)
    runtime.test_runtime()
