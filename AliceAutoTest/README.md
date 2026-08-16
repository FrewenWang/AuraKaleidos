# Phoenix 自动化测试框架（AliceAutoTest）

AliceAutoTest 是 Phoenix 教育客户端的历史自动化跑课工程，包含环境检测、HTTP/WebSocket
交互、课件组件执行、截图、日志、数据库和通知逻辑。离线工具层可以跨平台测试；完整跑课
依赖 Windows Phoenix 客户端、虚拟摄像头、授权账号、内部服务和数据库。

## 先看这里

> [!WARNING]
> `tools/run.py` 会清理进程、启动子进程、截图并尝试连接外部系统。不要在个人主机或未配置
> 的环境直接运行。仓库历史配置和源码中存在疑似真实凭据，必须先轮换并迁移到环境变量或
> Git 忽略的本地配置。

## 快速开始

推荐 Python 3.10–3.12。先只运行不需要 GUI、数据库或公网的离线测试：

```bash
cd AliceAutoTest
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r config/requirements-dev.txt
python -m pytest -q
```

完整依赖包含 PyQt5、OpenCV、gevent、MySQL 和 WebSocket 客户端：

```bash
python -m pip install -r config/requirements.txt
python tools/setup.py --test
```

`tools/setup.py` 默认会联网安装依赖并创建运行目录；如已手动安装依赖，仅执行 `--test`。

## 完整业务入口

仅在完成凭据治理、配置校验并准备好 Windows 客户端后运行：

```powershell
cd AliceAutoTest
python tools/run.py
```

macOS/Linux 只能验证平台兼容层和无设备工具，不能替代 Windows Phoenix 业务验收。

## 工程结构

```text
AliceAutoTest/
├── config/                 # 依赖、策略和本地运行配置
├── src/
│   ├── baseinfo/           # HTTP、WebSocket、课程与客户端信息
│   ├── config/             # 平台、日志、数据库、截图和进程封装
│   ├── fileconfig/         # JSON/TXT 文件读写
│   ├── modules/            # 跑课业务编排
│   └── settings.py         # 历史全局设置
├── tools/                  # 安装、启动、超时和虚拟摄像头工具
├── samples/                # 使用示例
├── tests/                  # 无外部副作用的 pytest 测试
├── docs/                   # 架构、配置、平台和 API 文档
├── pytest.ini
└── setup.py
```

## 文档

- [系统架构](docs/architecture.md)
- [配置与凭据治理](docs/configuration.md)
- [平台支持边界](docs/platform-support.md)
- [核心模块 API](docs/api-reference.md)

## 已知限制

- 历史入口仍存在固定 Windows 路径、私有服务和模块路径不一致，完整流程尚未达到开箱即用。
- `config/config.ini`、`src/settings.py` 不应继续保存真实账号、数据库或通知令牌。
- `tools/VCamTestTool/` 含 Windows 可执行文件，只能在受控 Windows 测试机上使用。
- 运行日志、截图、报告、临时 PID 和客户端资源不应提交 Git。
