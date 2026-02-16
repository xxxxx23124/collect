from functools import cache

class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        n = len(nums)
        @cache
        def dfs(val, idx):
            if idx == n:
                return val
            return dfs(val ^ nums[idx], idx + 1) + dfs(val, idx + 1)

        return dfs(0, 0)
