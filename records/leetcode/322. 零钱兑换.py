# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins_len = len(coins)
        dp = [[float("inf")] * (coins_len + 1) for _ in range(amount + 1)]
        for i in range(coins_len + 1):
            dp[0][i] = 0
        for i in range(1, amount + 1):
            for j in range(1, coins_len + 1):
                dp[i][j] = dp[i][j - 1]
                coin = coins[j - 1]
                if coin <= i and dp[i - coin][j] != float("inf"):
                    dp[i][j] = min(dp[i][j], dp[i - coin][j] + 1)

        return dp[-1][-1] if dp[-1][-1] != float("inf") else -1
