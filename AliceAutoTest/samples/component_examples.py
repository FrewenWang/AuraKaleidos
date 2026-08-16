#!/usr/bin/env python
"""
Phoenix自动化测试框架 - 组件使用示例

本示例展示如何使用Phoenix的业务组件：
1. 英语组件 (PxEn)
2. 数学组件 (PxMath)
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_english_component():
    """示例1: 英语组件使用"""
    print("=" * 60)
    print("示例1: 英语组件 (PxEn)")
    print("=" * 60)

    try:
        from src.modules.assembly.px_en import PxEn

        # 创建英语组件实例
        PxEn()
        print("✓ 英语组件实例化成功")

        # 组件提供的方法（示例）
        # 注意：实际使用需要有效的token和classid
        print("\n英语组件主要功能:")
        print("  - handle_countdown: 处理倒计时组件")
        print("  - en_handle_onetoone: 处理1V1问答组件")
        print("  - handle_showpicture_sendimage: 处理看图展示")
        print("  - role_paly_repeat: 处理角色扮演")
        print("  - magic_hot: 处理魔法帽组件")

        print("\n✓ 英语组件演示成功\n")

    except ImportError as e:
        print(f"✗ 导入失败: {e}\n")


def example_math_component():
    """示例2: 数学组件使用"""
    print("=" * 60)
    print("示例2: 数学组件 (PxMath)")
    print("=" * 60)

    try:
        from src.modules.assembly.px_math import PxMath

        # 创建数学组件实例
        PxMath()
        print("✓ 数学组件实例化成功")

        # 组件提供的方法（示例）
        print("\n数学组件主要功能:")
        print("  - question_answer: 处理做题投屏")
        print("  - many_ask: 处理多人问答")
        print("  - quick_qa: 处理快速点名")
        print("  - egg_play: 处理彩蛋")
        print("  - get_student_goal_by_type_id: 获取学生目标")
        print("  - set_student_medals: 设置学生勋章")

        print("\n✓ 数学组件演示成功\n")

    except ImportError as e:
        print(f"✗ 导入失败: {e}\n")


def example_assembly_module():
    """示例3: 学科组件包"""
    print("=" * 60)
    print("示例3: 学科组件包 (modules.assembly)")
    print("=" * 60)

    try:
        # 导入包
        from src.modules.assembly import px_en, px_math

        print("✓ 学科组件包导入成功")
        print(f"  - 英语模块: {px_en.__name__}")
        print(f"  - 数学模块: {px_math.__name__}")

        # 查看模块中的类
        print("\n英语模块中的类:")
        for name in dir(px_en):
            if not name.startswith("_"):
                obj = getattr(px_en, name)
                if isinstance(obj, type):
                    print(f"  - {name}")

        print("\n数学模块中的类:")
        for name in dir(px_math):
            if not name.startswith("_"):
                obj = getattr(px_math, name)
                if isinstance(obj, type):
                    print(f"  - {name}")

        print("\n✓ 学科组件包演示成功\n")

    except ImportError as e:
        print(f"✗ 导入失败: {e}\n")


def example_complete_workflow():
    """示例4: 完整工作流程"""
    print("=" * 60)
    print("示例4: 完整工作流程示例")
    print("=" * 60)

    print("""
完整的Phoenix自动化测试工作流程：

1. 初始化阶段
   - 读取配置 (src.config.base_config)
   - 初始化日志 (src.config.config_logging)
   - 获取硬件信息 (src.config.config_hardware)

2. 登录阶段
   - iPad端登录 (src.baseinfo.base_info)
   - 获取课程信息
   - WebSocket连接 (src.baseinfo.base_websocket)

3. 课中交互
   - iPad端初始化
   - 头像绑定
   - 组件交互
     - 英语组件 (modules.assembly.px_en)
     - 数学组件 (modules.assembly.px_math)

4. 课后处理
   - 下课操作
   - 数据上报
   - 结果统计

示例代码:
    from src.config.config_logging import ConfigLogging
    from src.baseinfo.base_info import BaseInfo
    from modules.assembly.px_en import PxEn
    from modules.assembly.px_math import PxMath

    # 初始化
    logger = ConfigLogging().write_logging()
    base_info = BaseInfo()

    # 登录
    login_data = base_info.ipad_login()

    # 组件测试
    px_en = PxEn()
    px_math = PxMath()
    """)

    print("✓ 工作流程示例演示成功\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  Phoenix自动化测试框架 - 组件使用示例")
    print("=" * 60 + "\n")

    try:
        # 运行各个示例
        example_english_component()
        example_math_component()
        example_assembly_module()
        example_complete_workflow()

        print("=" * 60)
        print("  所有示例运行成功！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
