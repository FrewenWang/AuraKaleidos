"""
43. 字符串相乘
https://leetcode.cn/problems/multiply-strings/description/

给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。



示例 1:

输入: num1 = "2", num2 = "3"
输出: "6"
示例 2:

输入: num1 = "123", num2 = "456"
输出: "56088"


提示：

1 <= num1.length, num2.length <= 200
num1 和 num2 只能由数字组成。
num1 和 num2 都不包含任何前导零，除了数字0本身。

"""
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if num1 == "0" or num2 == "0":
            return "0"
        # 结果为0
        ans = "0"
        m, n = len(num1), len(num2)  # 分别计算字符串1和字符串2的长度

        # range(start, stop, step) 函数
        # start: 循环的起始值（包含）。
        # stop: 循环的结束值（不包含）。
        # step: 步长（每次循环时递增或递减的值）。
        # 参数具体含义
        # n - 1 循环的起始值。n 是一个变量，因此 n - 1 是循环开始的数字。
        # -1：循环的结束值（不包含）。这个循环将从 n - 1 递减到 0，因为循环在到达 -1 时结束。
        #  -1：每次循环 i 的值递减 1。
        for i in range(n - 1, -1, -1):
            add = 0
            y = int(num2[i])
            curr = ["0"] * (n - i - 1)
            for j in range(m - 1, -1, -1):
                product = int(num1[j]) * y + add
                curr.append(str(product % 10))
                add = product // 10
            if add > 0:
                curr.append(str(add))
            curr = "".join(curr[::-1])
            ans = self.addStrings(ans, curr)

        return ans

    def addStrings(self, num1: str, num2: str) -> str:
        i, j = len(num1) - 1, len(num2) - 1
        add = 0
        ans = list()
        while i >= 0 or j >= 0 or add != 0:
            x = int(num1[i]) if i >= 0 else 0
            y = int(num2[j]) if j >= 0 else 0
            result = x + y + add
            ans.append(str(result % 10))
            add = result // 10
            i -= 1
            j -= 1
        return "".join(ans[::-1])


if __name__ == "__main__":
    s = Solution()
    result = s.multiply("12334", "243465456")
    print(result)
