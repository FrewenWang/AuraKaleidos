# AliceAutoTest 系统架构

## 处理链路

```text
tools/run.py
  ├── platform_compat / settings       环境与路径
  ├── ConfigLogging / ConfigScreen     日志与截图
  ├── ProcessManager                   清理 Phoenix/虚拟摄像头
  ├── Windows Phoenix/VCam             启动外部客户端
  └── PxPtAuto
       ├── HTTP/WebSocket              登录、课程和状态交互
       ├── px_run                      英语/数学组件跑课
       └── 报告、数据库、通知          结果归档与外部系统
```

## 分层

| 层 | 目录 | 职责 |
|---|---|---|
| 入口/编排 | `tools/` | 环境准备、进程生命周期和业务启动 |
| 业务 | `src/modules/`, `src/baseinfo/` | 课程信息、WebSocket 和组件流程 |
| 基础设施 | `src/config/`, `src/fileconfig/` | 平台、路径、日志、数据库、截图和文件读写 |
| 配置 | `config/`, `src/settings.py` | 历史运行参数和路径 |
| 验证 | `tests/`, `samples/` | 离线断言测试和人工示例 |

## 进程模型

完整入口会创建截图、超时控制、WebSocket 和虚拟摄像头相关进程。Windows 使用
`multiprocessing.freeze_support()`；退出和异常路径必须回收子进程。任何新增测试都不应
默认启动这些外部进程。

## 测试边界

`tests/` 只覆盖平台兼容、配置读取、文件工具和硬件信息等无外部副作用逻辑。以下属于环境
验收，不应伪装成普通单元测试：Phoenix GUI、虚拟摄像头、真实账号登录、内部 API、MySQL、
钉钉通知和完整三小时跑课。
