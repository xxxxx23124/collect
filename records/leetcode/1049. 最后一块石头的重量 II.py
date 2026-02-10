# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        stones_sum = sum(stones)
        stones_sum_half = stones_sum // 2
        stones_len = len(stones)
        dp = [[False] * (stones_len + 1) for _ in range(stones_sum_half + 1)]

        for i in range(stones_len + 1):
            dp[0][i] = True

        for i in range(1, stones_sum_half + 1):
            for j in range(1, stones_len + 1):
                if i >= stones[j - 1]:
                    dp[i][j] = dp[i][j - 1] or dp[i - stones[j - 1]][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        res = float("inf")
        for i in range(stones_sum_half, -1, -1):
            if dp[i][-1]:
                res = stones_sum - 2 * i
                break
        return res
