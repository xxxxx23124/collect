"""
1458. 两个子序列的最大点积
给你两个数组 nums1 和 nums2 。

请你返回 nums1 和 nums2 中两个长度相同的 非空 子序列的最大点积。

数组的非空子序列是通过删除原数组中某些元素（可能一个也不删除）后剩余数字组成的序列，但不能改变数字间相对顺序。比方说，[2,3,5] 是 [1,2,3,4,5] 的一个子序列而 [1,5,3] 不是。

示例 1：
输入：nums1 = [2,1,-2,5], nums2 = [3,0,-6]
输出：18
解释：从 nums1 中得到子序列 [2,-2] ，从 nums2 中得到子序列 [3,-6] 。
它们的点积为 (2*3 + (-2)*(-6)) = 18 。
示例 2：
输入：nums1 = [3,-2], nums2 = [2,-6,7]
输出：21
解释：从 nums1 中得到子序列 [3] ，从 nums2 中得到子序列 [7] 。
它们的点积为 (3*7) = 21 。
示例 3：
输入：nums1 = [-1,-1], nums2 = [1,1]
输出：-1
解释：从 nums1 中得到子序列 [-1] ，从 nums2 中得到子序列 [1] 。
它们的点积为 -1 。
提示：
1 <= nums1.length, nums2.length <= 500
-1000 <= nums1[i], nums2[i] <= 1000
"""

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
