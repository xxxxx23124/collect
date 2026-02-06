# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        strs_len = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for __ in range(strs_len + 1)]

        def count(s: str) -> tuple[int, int]:
            zeros = 0
            ones = 0
            for c in s:
                if c == "1":
                    ones += 1
                else:
                    zeros += 1
            return (zeros, ones)

        for i in range(1, strs_len + 1):
            for j in range(m + 1):
                for k in range(n + 1):
                    zeros, ones = count(strs[i - 1])
                    if zeros <= j and ones <= k:
                        dp[i][j][k] = max(
                            dp[i - 1][j][k], dp[i - 1][j - zeros][k - ones] + 1
                        )
                    else:
                        dp[i][j][k] = dp[i - 1][j][k]

        return dp[-1][-1][-1]