### 内存监测脚本

#### `memory_monitor.sh`
运行在手机端的内存监测脚本，运行结束后会自动在 `/data/local/tmp` 下创建以`YY-MM-DD-HH-MM` 命名的文件夹, 内部保存相关内存指标
```sh
Usage: ./memory_monitor.sh [options]
Options:
  -p, --pid       <proc_id>    PID of process to be sampled
  -n, --name      <proc_name>  Name of process to be sampled
  -i, --interval  <interval>   Period of sampling (default: 0.5)
  -h, --help                   Show this help
  -d, --dir                    Report store dir, default use date
```

Examples:
```sh
./memory_monitor.sh -p 12306          # Record process 12306's memory usage
./memory_monitor.sh -n aura_test_main # Record process `aura_test_main`'s memory usage
./memory_monitor.sh -p 12306 -i 0.1   # Record with periord 0.1s for dma_buf/gpu_mem and 0.1s * 2 = 0.2s with dumpsys meminfo
./memory_monitor.sh -p 12306 -d /data/local/tmp/my_dir # Record reports to custom dir
```

#### `plot_mem_info.py`
基于 matplotlib 实现的内存指标可视化，需要将手机端文件夹 pull 到电脑端才可以使用
```sh
Usage: plot_mem.py <dir_name> # 电脑端路径
```

> 需要 python3 环境和 matplotlib

#### 内存指标文档

- [原理介绍-飞书文档](https://xiaomi.f.mioffice.cn/docx/doxk47A173KbEljwPs6pRyvPVBe)
- [样例图-飞书文档](https://xiaomi.f.mioffice.cn/docx/doxk4EjShTsjP9wR9uWGRoQFHKd)
- 如对脚本内容实现不熟悉，可以使用 ChatGPT 进行解释

