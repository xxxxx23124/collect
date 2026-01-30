# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def jump(self, nums: List[int]) -> int:
        nums_len = len(nums)
        dp = [sys.maxsize] * nums_len
        dp[0] = 0
        for i in range(nums_len):
            if nums[i]:
                for j in range(nums[i]):
                    if i + j + 1 < nums_len:
                        dp[i + j + 1] = min(dp[i + j + 1], dp[i] + 1)
                    else:
                        dp[-1] = min(dp[-1], dp[i] + 1)
        return dp[-1]