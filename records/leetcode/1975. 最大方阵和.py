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