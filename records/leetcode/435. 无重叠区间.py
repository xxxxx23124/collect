# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 0:
            return 0
        intervals.sort(key=lambda x: x[1])
        n = len(intervals)
        right = intervals[0][1]
        res = 1

        for i in range(1, n):
            if intervals[i][0] >= right:
                res += 1
                right = intervals[i][1]

        return n - res
