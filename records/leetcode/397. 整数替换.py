# Author: https://leetcode.cn/u/233-zu/

from functools import cache


class Solution:
    @cache
    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0
        if n % 2:
            left = self.integerReplacement(n - 1)
            right = self.integerReplacement(n + 1)
            return min(left, right) + 1
        else:
            return 1 + self.integerReplacement(n // 2)
