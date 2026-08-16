# AliceAutoTest 平台支持

## 支持矩阵

| 能力 | Windows | macOS | Linux |
|---|---:|---:|---:|
| 离线 pytest | 支持 | 支持 | 支持 |
| 路径、日志、文件工具 | 支持 | 支持 | 支持 |
| Phoenix 学生端 | 支持 | 不支持 | 不支持 |
| VCamTestTool | 支持 | 不支持 | 不支持 |
| 完整自动化跑课 | 需业务环境 | 不支持 | 不支持 |

## 路径策略

`PlatformCompat` 使用 `Path.home()`、`APPDATA`、`~/Library/Application Support`、
`XDG_CONFIG_HOME` 和 `tempfile.gettempdir()` 处理通用路径。业务层仍有历史固定 Windows
路径，完整跨平台前必须逐项参数化。

## 平台注意事项

- Windows：完整流程可能需要管理员权限、Phoenix 客户端、批处理脚本和虚拟摄像头驱动。
- macOS：可运行离线测试；GUI/摄像头行为不能代表 Windows 业务结果。
- Linux：可运行离线测试；桌面环境、字体和 OpenCV GUI 依发行版而异。
- 所有平台：进程清理会按名称终止进程，执行完整入口前确认不会误伤其他程序。

## 验证命令

```bash
cd AliceAutoTest
python -m pytest -q
python tools/setup.py --test
```

第二条命令会创建少量临时目录，但不应启动 Phoenix 或连接业务服务。
