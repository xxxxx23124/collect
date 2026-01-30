# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        s_len = len(s)
        p_len = len(p)

        dp = [[False] * (p_len + 1) for _ in range(s_len + 1)]
        dp[0][0] = True

        def match(i: int, j: int) -> bool:
            if i == 0:
                return False

            if p[j - 1] == '.':
                return True
            else:
                return s[i - 1] == p[j - 1]

        for i in range(s_len + 1):
            for j in range(1, p_len + 1):
                if p[j - 1] == '*':
                    dp[i][j] |= dp[i][j - 2]
                    if match(i, j - 1):
                        dp[i][j] |= dp[i - 1][j]
                else:
                    if match(i, j):
                        dp[i][j] |= dp[i - 1][j - 1]

        return dp[s_len][p_len]
