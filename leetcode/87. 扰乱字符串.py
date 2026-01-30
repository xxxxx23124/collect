# Author: https://leetcode.cn/u/233-zu/

from functools import cache

class Solution_2026_01_30:
    def isScramble(self, s1: str, s2: str) -> bool:
        @cache
        def dfs(s1: str, s2: str) -> bool:
            if s1 == s2:
                return True
            if sorted(s1) != sorted(s2):
                return False
            n = len(s1)
            for i in range(1, n):
                if dfs(s1[:i], s2[:i]) and dfs(s1[i:], s2[i:]):
                    return True
                if dfs(s1[:i], s2[n - i :]) and dfs(s1[i:], s2[: n - i]):
                    return True
            return False

        return dfs(s1, s2)
