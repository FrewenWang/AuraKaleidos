"""
全局设置 - 跨平台配置
使用pathlib.Path统一处理路径，最小化平台差异
"""

import os
from pathlib import Path

# 当直接运行此脚本时，确保项目根目录在sys.path中
if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.platform_compat import get_platform, is_windows

# 初始化平台
platform = get_platform()

# 版本匹配规则
NO_AIB = "2.2.*.*"  # 非AIB版本
AIB_RULE = "2.1.*.*"  # AIB版本

# ==================== 跨平台路径配置 ====================

# Phoenix基础路径
if is_windows():
    P_P = Path(r"C:\phoenix")
    PHOENIX_BIN = Path(r"C:\Users\Admin\AppData\Roaming\phoenix\bin")
    SAVE_COURSE = Path(r"C:\Users\Admin\AppData\Roaming\phoenix\Resource")
    V_J = Path(r"C:\Users\Admin\AppData\Roaming\phoenix\Resource\component")
    SAVE_COMP = Path(r"C:\Users\Admin\AppData\Roaming\phoenixLocalComps\\")
    C_P = Path(r"C:\Users\Admin\AppData\Roaming\phoenixLocalComps\component")
    BACKUP_PATH = Path(r"C:\backup")
    LOG_PATH = Path(r"C:\Users\Admin\AppData\Local\Temp\AIBrowser\logs")
else:
    # macOS/Linux: 使用用户目录下的路径
    home = Path.home()
    app_data = Path(platform.get_app_data_path("Phoenix"))
    P_P = app_data
    PHOENIX_BIN = app_data / "bin"
    SAVE_COURSE = app_data / "Resource"
    V_J = app_data / "Resource" / "component"
    SAVE_COMP = app_data / "LocalComps"
    C_P = app_data / "LocalComps" / "component"
    BACKUP_PATH = home / "backup"
    LOG_PATH = home / ".cache" / "AIBrowser" / "logs"

# 日志路径
M_L = (
    Path(P_P).parent / "module_log"
    if is_windows()
    else Path(platform.get_temp_path())
)
M_L_I = M_L / "json.log"
ERR_L = (
    Path(P_P).parent / "errorLog"
    if is_windows()
    else Path(platform.get_temp_path()) / "errorLog"
)
STATUS_LOG = Path(P_P).parent / "logs" if is_windows() else LOG_PATH

# PID文件路径
PID_DIR = (
    Path(P_P).parent / "run"
    if is_windows()
    else Path(platform.get_temp_path()) / "run"
)
PID_PATH = PID_DIR / "run_phoenix.pid"
RESTART_LOG = (
    Path(P_P).parent / "restart_logs" / "restart_log.txt"
    if is_windows()
    else Path(platform.get_temp_path()) / "restart_log.txt"
)

# 课件数据路径
C_D = (
    Path(P_P).parent / "courseware_data"
    if is_windows()
    else Path(platform.get_app_data_path()) / "courseware_data"
)
COURSEWARE_DATA = C_D / "courseware_data.txt"
COURSEWARE_DOWNLOAD = C_D / "download.txt"
ERR_MD5 = C_D / "err_md5.txt"

# 输出路径
AUTO_OUTPUT = (
    Path(P_P).parent / "auto_output"
    if is_windows()
    else Path(platform.get_app_data_path()) / "auto_output"
)

# 备份时间（秒）
TIME_LINE = 86400

# PID进程名称
PID_NAME = "Phoenix.exe" if is_windows() else "Phoenix"

# 超时时间（秒）
TIME = 10800  # 3小时

# ==================== URL配置 ====================

COMP_URL = "http://phoenix.100tal.com/courseCheck/action/autoCheck/getComponentResource"
CHANGE_STATUS_URL = (
    r"http://phoenix.100tal.com/courseCheck/action/autoCheck/syncCheckStatus"
)
COURSE_COUNT = r"http://phoenix.100tal.com/courseCheck/action/autoCheck/getCoursewareCheckResults"
GET_COURSE_INFO = "http://phoenix.100tal.com/courseCheck/action/autoCheck/getCoursewareResource"
URL_COURSEWARE = "http://phoenix.100tal.com/courseCheck/action/autoCheck/getPendCheckCoursewares"
CHANGE_ONE_STATUS_URL = r"http://phoenix.100tal.com/courseCheck/action/autoCheck/syncCoursewareCheckResult"

# ==================== 钉钉机器人配置 ====================

ERROR_COURSE = os.getenv("ALICE_AUTOTEST_ERROR_WEBHOOK", "")
START_COURSE = os.getenv("ALICE_AUTOTEST_START_WEBHOOK", "")

# ==================== 数据库配置 ====================
HOST_MYSQL = os.getenv("ALICE_AUTOTEST_DB_HOST", "127.0.0.1")
PORT_MYSQL = int(os.getenv("ALICE_AUTOTEST_DB_PORT", "3306"))
MYSQL_UNAME = os.getenv("ALICE_AUTOTEST_DB_USER", "")
MYSQL_PWD = os.getenv("ALICE_AUTOTEST_DB_PASSWORD", "")
DATABASE_MYSQL = os.getenv("ALICE_AUTOTEST_DB_NAME", "")

# ==================== 其他配置 ====================

MP4_LOG = (
    Path(P_P).parent / "mp4" / "error_info.txt"
    if is_windows()
    else Path(platform.get_temp_path()) / "mp4" / "error_info.txt"
)
MP4_len = (
    Path(P_P).parent / "mp4" / "video_length.txt"
    if is_windows()
    else Path(platform.get_temp_path()) / "mp4" / "video_length.txt"
)
VIDEO_PATH = PHOENIX_BIN / "Resource" / "video"
COURSE_START_TIME = C_D / "courseInfo.txt"
PID_YES_OR_NO = PID_DIR

# ==================== 脚本路径 ====================

if is_windows():
    home = Path(r"C:\px_pt_auto")
    PX_RUN = home / "px_run.py"
    DOWN_MODUL = Path(r"C:\shaokai_test\module.py")
    KILL_PROCESS = home / "BaseConfig" / "kill_process.py"
    NUM_PATH = home / "BaseConfig" / "config.ini"
    STATUS_PATH = Path(r"C:\shaokai_test\status_info.ini")
    SHAOKAI_DINGDING = Path(r"C:\shaokai_test\shaokai_dingding.py")
else:
    home = Path.home() / "px_pt_auto"
    PX_RUN = home / "px_run.py"
    DOWN_MODUL = Path.home() / "shaokai_test" / "module.py"
    KILL_PROCESS = home / "BaseConfig" / "kill_process.py"
    NUM_PATH = home / "BaseConfig" / "config.ini"
    STATUS_PATH = Path.home() / "shaokai_test" / "status_info.ini"
    SHAOKAI_DINGDING = Path.home() / "shaokai_test" / "shaokai_dingding.py"


# ==================== 辅助函数 ====================


def ensure_directories():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        M_L,
        ERR_L,
        C_D,
        AUTO_OUTPUT,
        PID_DIR,
        BACKUP_PATH,
        LOG_PATH,
        STATUS_LOG,
        MP4_LOG.parent,
        VIDEO_PATH,
    ]

    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"创建目录失败 {dir_path}: {e}")


def get_status_info() -> dict:
    """获取当前平台状态信息"""
    return {
        "os": platform.system,
        "is_windows": is_windows(),
        "user_home": str(platform.get_user_home()),
        "app_data": str(platform.get_app_data_path()),
        "temp": platform.get_temp_path(),
        "hostname": platform.get_hostname(),
        "ip": platform.get_ip_address(),
        "mac": platform.get_mac_address(),
        "admin": platform.is_admin(),
    }


def print_platform_info():
    """打印平台信息"""
    info = get_status_info()
    print("=" * 50)
    print("平台信息:")
    print("=" * 50)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    print_platform_info()
    ensure_directories()
