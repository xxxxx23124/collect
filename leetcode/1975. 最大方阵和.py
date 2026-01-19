"""
给你一个 n x n 的整数方阵 matrix 。你可以执行以下操作 任意次 ：

选择 matrix 中 相邻 两个元素，并将它们都 乘以 -1 。
如果两个元素有 公共边 ，那么它们就是 相邻 的。

你的目的是 最大化 方阵元素的和。请你在执行以上操作之后，返回方阵的 最大 和。

示例 1：
输入：matrix = [[1,-1],[-1,1]]
输出：4
解释：我们可以执行以下操作使和等于 4 ：
- 将第一行的 2 个元素乘以 -1 。
- 将第一列的 2 个元素乘以 -1 。

示例 2：
输入：matrix = [[1,2,3],[-1,-2,-3],[1,2,3]]
输出：16
解释：我们可以执行以下操作使和等于 16 ：
- 将第二行的最后 2 个元素乘以 -1 。

n == matrix.length == matrix[i].length
2 <= n <= 250
-10^5 <= matrix[i][j] <= 10^5
"""

# Author: https://leetcode.cn/u/233-zu/
from typing import List

class Solution:
    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
        """
        核心：负号是可以移动的，具有传递性
        偶数个负数可以全消
        奇数个负数会剩下一个
        """

        total_sum = 0
        min_abs_val = float('inf')
        neg_count = 0
        for row in matrix:
            for val in row:
                abs_val = abs(val)
                total_sum += abs_val

                if abs_val < min_abs_val:
                    min_abs_val = abs_val

                if val < 0:
                    neg_count += 1

        if neg_count % 2 == 0:
            return total_sum
        else:
            # 2倍是因为 需要减去原先加上的本体
            return total_sum - 2 * min_abs_val