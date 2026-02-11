# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums_len = len(nums)
        half = (nums_len + 1) // 2
        rise = sorted(nums)
        j = half - 1
        k = nums_len - 1
        for i in range(0, nums_len, 2):
            nums[i] = rise[j]
            if i + 1 < nums_len:
                nums[i + 1] = rise[k]
            j -= 1
            k -= 1
