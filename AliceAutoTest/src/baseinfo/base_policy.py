#!/usr/bin/env python
# @Time    : 2019/10/15 18:39
# @Author  : liuqi
# @Site    :
# @File    : base_policy.py
# @Software: PyCharm
from pathlib import Path

import xlrd
from xlutils.copy import copy

# 策略文件路径 - 使用项目内相对路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
POLICY_FILE = str(_PROJECT_ROOT / "config" / "policy.xls")


class BasePolicy:
    def __init__(self):
        pass

    def read_excel(
        self, dv, h, sc, ec
    ):  # dv表示默认值，h表示当前行从第几列开始，sc表示当前行开始列，ec表示当前行最后列
        data = xlrd.open_workbook(POLICY_FILE)
        table = data.sheet_by_name("Sheet1")
        values = table.row_values(h, start_colx=sc, end_colx=ec)
        val = dv
        h1 = h
        sc1 = sc
        ec1 = ec
        rep = None
        print(values)
        for i in range(1, len(values)):
            if values[i] != "T" and i == len(values) - 1:
                for j in range(1, len(values)):
                    if values[j] == "F":
                        self.write_excel(h, j, "T")
                self.read_excel(val, h1, sc1, ec1)
            elif values[i] == "T":
                val = values[i + 1]
                print(val)
                self.write_excel(h, i, "F")
                break
            else:
                continue
        rep = val if isinstance(val, str) else int(val)
        return rep

    def write_excel(self, nrol, crol, value):
        data = xlrd.open_workbook(POLICY_FILE)
        wb = copy(data)
        table = wb.get_sheet("Sheet1")
        table.write(nrol, crol, value)
        try:
            wb.save(POLICY_FILE)
        except Exception as e:
            print("write_excel_error:", e)


if __name__ == "__main__":
    oto_list = {
        "index": 2,
        "result": 9,  # 6 结束 7 提示 1 正确
        "stuID": 3,
        "typeID": 3,
    }

    data = BasePolicy()
    policy = data.read_excel("Excellent", 3, 0, 7)
    print(policy)
    # policy = 0
    # if policy and policy != 7:
    #     oto_list['result'] = policy
    # elif policy and policy == 7:
    #     oto_list['result'] = policy
    #     print(oto_list)
    #     oto_list['result'] = 1
    # print(oto_list)
