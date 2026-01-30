# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        nums_len = len(nums)
        dp = [-sys.maxsize - 1] * (nums_len + 1)
        for i in range(1,nums_len+1):
            dp[i] = max(nums[i-1], nums[i-1] + dp[i-1])
        return max(dp)