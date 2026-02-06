# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        nums_len = len(nums)
        nums_sum = sum(nums)
        if nums_len < 2 or nums_sum % 2 == 1:
            return False
        half = nums_sum // 2
        dp = [[False] * (half + 1) for _ in range(nums_len)]
        for i in range(nums_len):
            dp[i][0] = True
        if nums[0] <= half:
            dp[0][nums[0]] = True
        for i in range(1, nums_len):
            for j in range(1, half + 1):
                if j < nums[i]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - nums[i]]
        return dp[-1][-1]
