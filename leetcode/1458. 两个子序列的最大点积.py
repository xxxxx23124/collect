# Author: https://leetcode.cn/u/233-zu/

from typing import List

# two solutions

# A 自底向上

class SolutionA:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        n, m = len(nums1), len(nums2)
        dp = [[float('-inf')] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                product = nums1[i] * nums2[j]
                current_max = product
                if i > 0 and j > 0:
                    current_max = max(current_max, product + dp[i-1][j-1])
                if i > 0:
                    current_max = max(current_max, dp[i-1][j])
                if j > 0:
                    current_max = max(current_max, dp[i][j-1])
                dp[i][j] = current_max
        return dp[n-1][m-1]

# B 自顶向下
import sys
from functools import cache

sys.setrecursionlimit(5000)


class SolutionB:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return float('-inf')
            product = nums1[i] * nums2[j]
            res = product
            prev = dfs(i - 1, j - 1)
            if prev != float('-inf'):
                res = max(res, product + prev)
            res = max(res, dfs(i - 1, j))
            res = max(res, dfs(i, j - 1))
            return res

        return dfs(len(nums1) - 1, len(nums2) - 1)
