#!/usr/bin/env python
"""
Phoenix自动化测试框架 - 项目配置
支持开发和安装模式
"""

from setuptools import find_packages, setup

setup(
    name="phoenix-auto-test",
    version="2.0.0",
    description="Phoenix教育软件自动化测试框架",
    author="Phoenix Team",
    packages=find_packages(where="src") + find_packages(where="modules"),
    package_dir={"": "src", "modules": "modules"},
    python_requires=">=3.6",
    install_requires=[
        "psutil>=5.6.2",
        "requests>=2.21.0",
        "PyMySQL>=0.9.3",
        "PyQt5>=5.12.1",
        "opencv-python>=4.1.1.26",
        "websocket-client>=0.57.0",
        "gevent>=1.4.0",
        "Pillow>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "phoenix-test=tools.run:main",
        ],
    },
)
