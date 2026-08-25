# AliceAutoTest 配置与凭据治理

## 安全要求

仓库历史文件中存在账号、数据库、通知 webhook、远程控制和内部服务字段。它们应视为已经
暴露：先在对应平台轮换或吊销，再进行代码迁移。不要仅删除 Git 当前版本后继续使用旧值，
因为提交历史仍可能保留内容。

目标配置层级：

```text
代码默认值（无密钥）
  < 环境变量
  < config/config.local.ini（Git 忽略）
```

建议提交 `config/config.example.ini`，只保留以下占位结构：

```ini
[DB]
host = <db-host>
username = <db-user>
password = <db-password>
port = 3306
database = <db-name>

[LOGIN]
en_username = <english-test-user>
en_password = <english-test-password>
ma_username = <math-test-user>
ma_password = <math-test-password>

[HTTP]
base_login_url = <login-url>
base_auth_url = <auth-url>
base_new_websocket_url = <websocket-url>

[REPORT_PATH]
logging_path = logs
reportpath = logs/reports

[DINGTALK]
release_pass_url =
release_error_url =
test_pass_url =
test_error_url =
error_course =
start_course =
user_right_info =
user_error_info =

[CONTACTS]
test_phone =
```

`ReadBaseConfig` 的通用环境变量格式为
`ALICE_AUTOTEST_<SECTION>_<KEY>`，例如
`ALICE_AUTOTEST_DB_HOST`、`ALICE_AUTOTEST_CONTACTS_TEST_PHONE`。为兼容历史部署，通知和
登录手机号还支持以下变量：

| 用途 | 环境变量 |
|---|---|
| 登录手机号 | `ALICE_AUTOTEST_PHONE` |
| 发布通过/失败通知 | `ALICE_AUTOTEST_DING_RELEASE_PASS` / `ALICE_AUTOTEST_DING_RELEASE_ERROR` |
| 测试通过/失败通知 | `ALICE_AUTOTEST_DING_TEST_PASS` / `ALICE_AUTOTEST_DING_TEST_ERROR` |
| 异常课通知 | `ALICE_AUTOTEST_DING_ERROR_COURSE` |
| 通过/失败联系人 | `ALICE_AUTOTEST_DING_SUCCESS_USERS` / `ALICE_AUTOTEST_DING_ERROR_USERS` |

通知未配置 webhook 时会记录警告并跳过网络请求，不会向空地址发起请求。

## 路径配置

新代码使用 `pathlib.Path`，优先从脚本位置、用户目录和系统临时目录派生路径。Phoenix 安装
目录、资源目录和虚拟摄像头位置应由环境变量或本地配置传入，不新增固定用户名、盘符或
作者机器目录。

## 配置变更检查

提交前至少确认：

- `git diff` 中没有密码、token、webhook、数据库地址或私有账号。
- 示例配置使用不可工作的占位值。
- 日志不会打印完整 token、密码或个人身份信息。
- 本地配置、数据库导出、截图和报告已被 `.gitignore` 排除。
- SQL 通过参数绑定传值；不要把业务字段拼接进查询字符串。
