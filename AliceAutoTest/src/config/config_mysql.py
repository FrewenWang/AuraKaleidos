"""MySQL 结果存储。

数据库连接只在显式实例化时建立，查询统一支持参数绑定，便于离线测试并避免
把业务字段拼接到 SQL 中。
"""

import json

import pymysql

from src.config.base_config import ReadBaseConfig
from src.config.config_logging import ConfigLogging


class MySqlConfig:
    def __init__(self, config=None, connect_factory=None):
        basedata = config or ReadBaseConfig()
        factory = connect_factory or pymysql.connect
        self.host = basedata.get_db("host")
        self.username = basedata.get_db("username")
        self.password = basedata.get_db("password")
        self.port = int(basedata.get_db("port"))
        self.dbname = basedata.get_db("database")
        self.connect = factory(
            host=self.host,
            user=self.username,
            password=self.password,
            database=self.dbname,
            port=self.port,
            charset="utf8mb4",
        )
        self.cursor = self.connect.cursor(cursor=pymysql.cursors.DictCursor)

    def endmysql(self):
        """幂等关闭游标和连接。"""
        cursor = getattr(self, "cursor", None)
        connection = getattr(self, "connect", None)
        if cursor is not None:
            cursor.close()
            self.cursor = None
        if connection is not None:
            connection.close()
            self.connect = None

    def readmysql(self, sql, params=None):
        self.cursor.execute(sql, params)
        return self.cursor.fetchall()

    def insertmysql(self, sql, params=None):
        try:
            result = self.cursor.execute(sql, params)
            self.connect.commit()
            return result
        except Exception:
            self.connect.rollback()
            raise

    def updatemysql(self, sql, params=None):
        try:
            self.cursor.execute(sql, params)
            self.connect.commit()
            return True
        except Exception:
            self.connect.rollback()
            return False
        finally:
            self.endmysql()


class MySqlHandler(MySqlConfig):
    def __init__(self, config=None, connect_factory=None):
        self.logger = ConfigLogging().write_logging()
        self.connected = False
        try:
            super().__init__(config=config, connect_factory=connect_factory)
            self.connected = True
        except Exception as exc:
            self.logger.error("数据库连接失败：%s", exc)

    def is_connected(self):
        return self.connected

    def handle_insert(self, content):
        result = {"code": None, "detail": "", "results": {}}
        if not self.connected:
            result.update(code=503, detail="database unavailable")
            return result

        sql = (
            "INSERT INTO courseware "
            "(courseware_id,courseware_name,result_code,result_name,error_code,"
            "error_detail,test_master_name,test_master_mac,pc_version,login_name,"
            "last_cid,remote_id,remote_pwd,assembly_count,create_date) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())"
        )
        params = (
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
            json.dumps(content["componentcount"], ensure_ascii=False),
        )

        try:
            self.logger.info(
                "写入课件结果：courseware_id=%s, classname=%s",
                content["classid"],
                content["classname"],
            )
            self.insertmysql(sql, params)
            rows = self.readmysql(
                "SELECT * FROM courseware cs WHERE cs.`courseware_id` = %s",
                (content["classid"],),
            )
            if rows:
                result.update(code=200, results=rows)
            else:
                result.update(code=201, detail="insert mysql error")
                self.logger.error(
                    "数据库写入后未查到记录：courseware_id=%s",
                    content["classid"],
                )
        except Exception as exc:
            result.update(code=500, detail=str(exc))
            self.logger.error("数据库插入失败：%s", exc)
        finally:
            self.endmysql()
            self.connected = False
        return result
