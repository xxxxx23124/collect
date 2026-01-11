"""
85. 最大矩形
给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
示例 1：
输入：matrix =
[["1","0","1","0","0"],
["1","0","1","1","1"],
["1","1","1","1","1"],
["1","0","0","1","0"]]
输出：6
解释：最大矩形如上图所示。
示例 2：
输入：matrix = [["0"]]
输出：0
示例 3：
输入：matrix = [["1"]]
输出：1
提示：
rows == matrix.length
cols == matrix[0].length
1 <= rows, cols <= 200
matrix[i][j] 为 '0' 或 '1'
"""

# Author: https://leetcode.cn/u/233-zu/
from typing import List

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0

        rows = len(matrix)
        cols = len(matrix[0])

        heights = [0] * cols
        max_area = 0

        for row in matrix:
            for i in range(cols):
                if row[i] == "1":
                    heights[i] += 1
                else:
                    heights[i] = 0

            max_area = max(max_area, self.largestRectangleArea(heights))
        return max_area

    def largestRectangleArea(self, heights: List[int]) -> int:
        tmp_heights = [0] + heights + [0]
        stack = []
        res = 0

        for i in range(len(tmp_heights)):
            while stack and tmp_heights[i] < tmp_heights[stack[-1]]:
                h = tmp_heights[stack.pop()]
                w = (i - 1) - stack[-1]
                res = max(res, h * w)
            stack.append(i)
        return res