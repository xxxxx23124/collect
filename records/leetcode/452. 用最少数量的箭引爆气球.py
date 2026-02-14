# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x: x[1])
        n = len(points)
        right = points[0][1]
        res = 1

        for i in range(1, n):
            if points[i][0] > right:
                res += 1
                right = points[i][1]

        return res
