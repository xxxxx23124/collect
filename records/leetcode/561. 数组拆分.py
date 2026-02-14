# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        res = 0
        for i in range(0, n, 2):
            res += min(nums[i], nums[i + 1])
        return res
