# Author: https://leetcode.cn/u/233-zu/

from functools import cache


class SolutionA:
    def numSquares(self, n: int) -> int:

        @cache
        def dfs(n: int):
            res = sys.maxsize
            i = 1
            while i**2 <= n:
                if n - i**2 > 0:
                    res = min(res, dfs(n - i**2))
                else:
                    return 1
                i += 1
            return res + 1

        return dfs(n)

class SolutionB:
    def numSquares(self, n: int) -> int:
        dp = [sys.maxsize] * (n + 1)
        dp[0] = 0
        squares = [i * i for i in range(1, int(n**0.5) + 1)]
        for i in range(1, n + 1):
            for square in squares:
                if square > i:
                    break
                dp[i] = min(dp[i], dp[i - square] + 1)

        return dp[n]



