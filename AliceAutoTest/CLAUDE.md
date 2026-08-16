# CLAUDE.md

本文件为Claude Code (claude.ai/code) 提供代码库工作指南。

## 项目概述

Phoenix自动化测试框架 - 用于Phoenix教育软件的自动化测试系统，支持英语和数学科目课件测试。

## 快速开始

```bash
# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 运行测试
python setup.py --test

# 启动程序
python run.py
```

## 系统要求

- **Python**: 3.6+
- **操作系统**: Windows, macOS, Linux

## 项目结构

```
AliceAutoTest/
├── src/                          # 框架代码（通用基础设施）
│   ├── __init__.py
│   ├── settings.py               # 全局设置和常量
│   ├── config/                   # 配置模块
│   │   ├── platform_compat.py    # 跨平台兼容层（统一API）
│   │   ├── base_config.py        # 基础配置读取
│   │   ├── config_hardware.py    # 硬件配置（使用psutil）
│   │   ├── config_logging.py     # 日志配置
│   │   ├── config_ding.py        # 钉钉通知
│   │   ├── config_mysql.py       # 数据库配置
│   │   ├── config_screen.py      # 截图工具
│   │   ├── kill_process.py       # 进程管理（跨平台）
│   │   ├── base_file.py          # 文件操作基类
│   │   └── datetime_utils.py     # 时间工具
│   ├── baseinfo/                 # 基础信息模块
│   │   ├── base_info.py          # API客户端
│   │   ├── base_websocket.py     # WebSocket客户端
│   │   ├── websocket_app.py
│   │   ├── pipe_config.py
│   │   ├── pc_info.py
│   │   ├── base_policy.py
│   │   ├── rename_file.py
│   │   └── start_stu_client.py
│   ├── fileconfig/               # 文件配置模块
│   │   ├── file_config_base.py   # 文件操作基类
│   │   ├── json_file_config.py   # JSON文件处理
│   │   └── txt_file_config.py    # 文本文件处理
│   └── utils/                    # 工具类
│       └── other/                # 其他工具
│           ├── change_status.py
│           ├── get_com_version.py
│           └── run_many_time.py
├── modules/                      # Phoenix业务代码
│   ├── __init__.py
│   ├── px_run.py                 # 主程序逻辑
│   ├── change.py                 # 状态变更
│   ├── assembly/                 # 学科组件
│   │   ├── px_en.py              # 英语组件
│   │   └── px_math.py            # 数学组件
│   └── tools/                    # 业务相关工具
│       ├── three_hours.py        # 超时策略
│       └── VCamTestTool/         # 虚拟摄像头工具
│           └── control2.py
├── samples/                      # 示例代码
│   ├── basic_usage.py            # 基础使用示例
│   └── component_examples.py     # 组件使用示例
├── config/                       # 配置文件目录
│   ├── config.ini                # 主配置文件
│   ├── requirements.txt          # 依赖列表
│   └── ...
├── tools/                        # 启动脚本
│   ├── run.py                    # 统一入口
│   └── setup.py                  # 安装脚本
└── setup.py                      # 项目安装配置
```

## 跨平台设计

### 核心原则

1. **统一API**: 使用 `config/platform_compat.py` 封装平台差异
2. **优先使用跨平台库**: `psutil`、`pathlib`、`subprocess`
3. **最小化条件分支**: 只在必要时使用 `is_windows()` 判断
4. **优雅降级**: 非核心功能在不受限时自动跳过

### 平台支持

| 功能 | Windows | macOS | Linux |
|------|---------|-------|-------|
| Phoenix学生端 | ✅ | ❌ | ❌ |
| 虚拟摄像头 | ✅ | ❌ | ❌ |
| API/WebSocket | ✅ | ✅ | ✅ |
| 数据库/钉钉 | ✅ | ✅ | ✅ |

### 使用跨平台API

```python
from src.config.platform_compat import get_platform

# 统一的API，无需关心底层平台
platform = get_platform()
path = platform.get_app_data_path()
```

## 常用命令

```bash
# 安装依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r config/requirements.txt

# 运行测试
python setup.py --test

# 启动程序
python tools/run.py

# 运行示例
python samples/basic_usage.py
python samples/component_examples.py
```

## 配置

编辑 `config/config.ini`：
- `[DB]`: 数据库连接信息
- `[LOGIN]`: 账号凭证
- `[HTTP]`: API端点配置

## 依赖库

核心依赖：PyQt5, OpenCV, psutil, requests, websocket-client, PyMySQL, gevent

## 代码组织原则

### 三层架构

1. **`src/` 框架层**：与业务无关的基础设施
   - 配置管理、日志记录、数据库操作
   - 文件操作、进程管理、跨平台兼容
   - 工具类、辅助函数

2. **`modules/` 业务层**：Phoenix测试相关业务逻辑
   - 主程序运行逻辑 (px_run.py)
   - 学科组件 (英语、数学)
   - 业务工具 (超时策略、虚拟摄像头)

3. **`samples/` 示例层**：展示框架使用方法
   - 基础使用示例
   - 组件使用示例

### 导入规范

```python
# 框架层内部导入
from src.config.base_config import ReadBaseConfig
from src.baseinfo.base_info import BaseInfo

# 业务层导入框架层
from src.config.config_logging import ConfigLogging
from src.fileconfig.json_file_config import HandleJson

# 业务层内部导入
from modules.assembly.px_en import PxEn
```

## 注意事项

- 所有路径使用 `pathlib.Path` 处理
- 进程管理使用 `psutil` 实现跨平台
- 命令执行使用 `subprocess` 替代 `os.system`
- 虚拟摄像头功能仅支持Windows
- 框架代码放在 `src/`，业务代码放在 `modules/`
