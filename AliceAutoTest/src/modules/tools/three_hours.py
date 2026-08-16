"""
三小时超时策略模块
"""

import sys
import time
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import settings
from src.config.config_logging import ConfigLogging

logger = ConfigLogging().write_logging()


def get_pid(pid_name):
    """获取进程PID"""
    for proc in __import__("psutil").process_iter(["pid", "name"]):
        try:
            if proc.info["name"] == pid_name:
                return proc.info["pid"]
        except (
            __import__("psutil").NoSuchProcess,
            __import__("psutil").AccessDenied,
        ):
            pass
    return None


def write_result(pass_or_fail, info):
    """写入结果"""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y_%m_%d")
    path = Path(settings.AUTO_OUTPUT) / f"{date_str}.json"

    try:
        import json

        if path.exists():
            with open(path, "r+", encoding="utf-8") as f:
                item = json.loads(f.read())
                item["total"] += 1
                item[pass_or_fail] += 1
                item[pass_or_fail + "_list"].append(info)
                f.seek(0)
                f.write(json.dumps(item, indent=2, ensure_ascii=False))
        else:
            item = {
                "total": 1,
                pass_or_fail: 1,
                "pass" if pass_or_fail == "fail" else "fail": 0,
                pass_or_fail + "_list": [info],
                "fail_list" if pass_or_fail == "pass" else "pass_list": [],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(item, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"写入结果失败: {e}")


def main():
    """主函数 - 三小时超时策略"""
    logger.info("启动三小时超时策略")

    start_time = time.time()
    timeout = settings.TIME  # 3小时

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.warning(f"测试超时 ({timeout}s)，强制退出")
            # 这里可以添加强制退出逻辑
            break

        # 检查是否需要继续运行
        pid = get_pid(settings.PID_NAME)
        if not pid:
            logger.info("目标进程已退出")
            break

        time.sleep(60)  # 每分钟检查一次


if __name__ == "__main__":
    main()
