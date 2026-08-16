# AliceAutoTest 核心模块 API

## 平台兼容层

```python
from src.config.platform_compat import get_platform

platform = get_platform()
home = platform.get_user_home()
app_data = platform.get_app_data_path("Phoenix")
temporary = platform.get_temp_path()
result = platform.check_compatibility()
```

`PlatformCompat` 还提供目录创建、文件复制、默认程序打开、命令执行和进程清理。进程清理是
有副作用操作，测试时应使用不存在的进程名或 mock。

## 配置读取

`src/config/base_config.py` 负责读取 INI 配置。调用方不应直接记录密码或 token；缺少配置时
应给出字段名和配置来源，不回显敏感值。

## 文件工具

`src/fileconfig/` 封装 JSON/TXT 文件读写。新接口优先接受 `Path`/路径参数，并显式指定
UTF-8；测试使用临时目录，不写用户真实配置目录。

## 日志与截图

- `ConfigLogging`：日志归档与运行记录。
- `ConfigScreen`：截图和历史截图清理。
- `PidConfig`、`ProcessManager`：PID、进程和外部工具管理。

这些模块会接触文件系统或操作系统进程，应由业务入口显式调用，避免在 import 阶段产生
副作用。

## 业务入口

`tools/run.py` 是完整业务编排入口，`src/modules/px_run.py` 负责课中组件流程。二者依赖
外部客户端和私有服务，不属于可离线复用的稳定 API。
