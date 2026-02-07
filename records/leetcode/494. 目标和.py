# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        nums_sum = sum(nums)

        if (nums_sum - target) < 0:
            return 0
        elif (nums_sum - target) % 2 == 1:
            return 0

        nums_len = len(nums)
        neg = (nums_sum - target) // 2
        dp = [[0] * (nums_len + 1) for _ in range(neg + 1)]
        dp[0][0] = 1

        for i in range(neg + 1):
            for j in range(1, nums_len + 1):
                if i - nums[j - 1] >= 0:
                    dp[i][j] = dp[i - nums[j - 1]][j - 1] + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        return dp[-1][-1]
