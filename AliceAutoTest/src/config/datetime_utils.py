import time


class ConfigTime:
    @property
    def reporttime(self):
        # 简单可读形式
        time.asctime(time.localtime(time.time()))
        # 格式化时间
        formatted_time = time.strftime(
            "%Y-%m-%d %H_%M_%S", time.localtime(time.time())
        )
        return formatted_time


if __name__ == "__main__":
    data = ConfigTime()
    print(data.reporttime)
    time.sleep(3)
    print(data.reporttime)
