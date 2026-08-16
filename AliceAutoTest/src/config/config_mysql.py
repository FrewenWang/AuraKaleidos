# 修复数据库存储失败问题，添加数据库存储日志
import contextlib
import json

import pymysql

from src.config.base_config import ReadBaseConfig

# Mysql数据库配置
from src.config.config_logging import ConfigLogging


class MySqlConfig:
    def __init__(self):
        basedata = ReadBaseConfig()
        self.host = basedata.get_db("host")
        self.username = basedata.get_db("username")
        self.password = basedata.get_db("password")
        self.port = basedata.get_db("port")
        self.dbname = basedata.get_db("database")
        # 连接数据库
        self.connect = pymysql.connect(
            self.host, self.username, self.password, self.dbname, charset="utf8"
        )
        # 获取操作游标
        self.cursor = self.connect.cursor(cursor=pymysql.cursors.DictCursor)

    # 数据库基础操作
    def endmysql(self):
        # 关闭游标
        self.cursor.close()
        # 关闭数据库链接
        self.connect.close()

    # 读取数据库内容
    def readmysql(self, sql):
        # 使用execute执行sql语句
        self.cursor.execute(sql)
        # 获取数据
        data = self.cursor.fetchall()
        return data

    # 插入数据内容
    def insertmysql(self, sql):
        try:
            # 执行sql语句
            result = self.cursor.execute(sql)
            # 提交到数据库执行
            self.connect.commit()
        except Exception as e:
            print("insert_error:", e)
            # 遇到错误时回滚
            self.connect.rollback()
        return str(result)

    # 更新数据库表数据【修改/删除】
    def updatemysql(self, sql):
        try:
            # 执行sql语句
            self.cursor.execute(sql)
            # 提交到数据库执行
            self.connect.commit()
        except Exception:  # 发生错误时回滚
            self.connect.rollback()
        # 关闭操作
        self.endmysql()
        return True


class MySqlHandler(MySqlConfig):
    def __init__(self):
        self.logger = ConfigLogging().write_logging()
        with contextlib.suppress(Exception):
            MySqlConfig.__init__(self)

    def handle_insert(self, content):
        result = {"code": None, "detail": "", "results": {}}
        # buffrt=json.loads(content['componentcount'])
        # # content['componentcount']=json.loads(buffer)
        # print(buffrt['en_grab-red-packet-bird_1.0.6'])
        try:
            sql = (
                " INSERT INTO courseware (courseware_id,courseware_name,result_code,result_name,error_code,error_detail,test_master_name,test_master_mac,pc_version,login_name,last_cid,remote_id,remote_pwd,assembly_count,create_date) "
                "VALUES ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}',now())".format(
                    content["classid"],
                    content["classname"],
                    content["result_code"],
                    content["result_name"],
                    content["error_code"],
                    content["error_detail"],
                    content["hostname"],
                    content["mac"],
                    content["px_version"],
                    content["username"],
                    content["cid"],
                    content["remote_id"],
                    content["remote_pwd"],
                    json.dumps(content["componentcount"]),
                )
            )
            self.logger.info("插入数据库数据：" + sql)
            result_name = self.insertmysql(sql)
            print("结果:" + result_name)
            sql_status = "SELECT * FROM courseware cs WHERE cs.`courseware_id` = '{}'".format(
                content["classid"]
            )
            read_sql = str(self.readmysql(sql_status))
            print("看这里", read_sql)
            if len(read_sql) != 0:
                self.logger.info("数据库插入成功：" + read_sql)
                result["code"] = 200
            else:
                result["code"] = 201
                result["detail"] = "insert mysql error"
                self.logger.error("数据库插入失败：" + read_sql)
        except Exception as e:
            self.logger.error("数据库插入失败：具体看content参数数据" + str(e))
            print("insert_error:", e)
        finally:
            self.endmysql()
