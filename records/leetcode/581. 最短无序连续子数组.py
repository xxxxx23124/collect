# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        right = -1
        left = -1
        right_num = -float("inf")
        left_num = float("inf")
        for i in range(n):
            if right_num > nums[i]:
                right = i
            else:
                right_num = nums[i]
        for i in range(n - 1, -1, -1):
            if left_num < nums[i]:
                left = i
            else:
                left_num = nums[i]
        return 0 if right == -1 else right - left + 1
