# Author: https://leetcode.cn/u/233-zu/

from typing import List

class Solution_2026_01_11:
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

class Solution_2026_01_30:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        heights = [0] * n
        max_rec = 0
        for row in matrix:
            for i in range(n):
                if row[i] == '1':
                    heights[i] += 1
                else:
                    heights[i] = 0
            max_rec = max(max_rec, self.monotonicIncreasingStack(heights))
        return max_rec

    def monotonicIncreasingStack(self, heights: List[int]) -> int:
        tmp_heights = [0] + heights + [0]
        tmp_heights_len = len(tmp_heights)
        stack = [0]
        max_rec = 0
        for i in range(1, tmp_heights_len):
            while tmp_heights[i] < tmp_heights[stack[-1]]:
                h = tmp_heights[stack.pop()]
                w = (i - 1) - stack[-1]
                max_rec = max(max_rec, h * w)
            stack.append(i)
        return max_rec
