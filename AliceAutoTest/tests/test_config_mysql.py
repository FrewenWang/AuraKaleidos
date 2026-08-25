"""MySQL 参数绑定、事务和离线降级测试。"""

import logging

from src.config import config_mysql


class FakeConfig:
    values = {
        "host": "db.test",
        "username": "tester",
        "password": "secret",
        "port": "3307",
        "database": "phoenix",
    }

    def get_db(self, name):
        return self.values[name]


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executions = []
        self.closed = False

    def execute(self, sql, params=None):
        self.executions.append((sql, params))
        return 1

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, rows=None):
        self.fake_cursor = FakeCursor(rows)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self, **_kwargs):
        return self.fake_cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def _disable_file_logging(monkeypatch):
    logger = logging.getLogger("test-config-mysql")
    monkeypatch.setattr(
        config_mysql.ConfigLogging,
        "write_logging",
        lambda _self: logger,
    )


def test_connection_uses_cross_platform_keyword_arguments():
    connection = FakeConnection()
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return connection

    database = config_mysql.MySqlConfig(FakeConfig(), factory)

    assert received == {
        "host": "db.test",
        "user": "tester",
        "password": "secret",
        "database": "phoenix",
        "port": 3307,
        "charset": "utf8mb4",
    }
    database.endmysql()
    database.endmysql()
    assert connection.closed
    assert connection.fake_cursor.closed


def test_handler_uses_parameterized_queries(monkeypatch):
    _disable_file_logging(monkeypatch)
    connection = FakeConnection(rows=[{"courseware_id": "course-1"}])
    handler = config_mysql.MySqlHandler(
        FakeConfig(), lambda **_kwargs: connection
    )
    dangerous_name = "name'); DROP TABLE courseware; --"
    content = {
        "classid": "course-1",
        "classname": dangerous_name,
        "result_code": 0,
        "result_name": "通过",
        "error_code": "",
        "error_detail": "",
        "hostname": "host",
        "mac": "00:11:22:33:44:55",
        "px_version": "1.0",
        "username": "tester",
        "cid": "cid",
        "remote_id": "remote",
        "remote_pwd": "password",
        "componentcount": {"video": 1},
    }

    result = handler.handle_insert(content)

    insert_sql, insert_params = connection.fake_cursor.executions[0]
    select_sql, select_params = connection.fake_cursor.executions[1]
    assert dangerous_name not in insert_sql
    assert dangerous_name in insert_params
    assert "%s" in insert_sql
    assert select_params == ("course-1",)
    assert "course-1" not in select_sql
    assert result["code"] == 200
    assert connection.commits == 1
    assert connection.closed


def test_connection_failure_is_observable(monkeypatch):
    _disable_file_logging(monkeypatch)

    def fail_connect(**_kwargs):
        raise OSError("database offline")

    handler = config_mysql.MySqlHandler(FakeConfig(), fail_connect)

    assert not handler.is_connected()
    assert handler.handle_insert({}) == {
        "code": 503,
        "detail": "database unavailable",
        "results": {},
    }
