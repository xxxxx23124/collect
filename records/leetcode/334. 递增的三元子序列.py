# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = nums[0]
        second = float("inf")
        for num in nums:
            if num > second:
                return True
            elif num > first:
                second = num
            else:
                first = num
        return False
