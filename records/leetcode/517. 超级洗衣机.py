# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def findMinMoves(self, machines: List[int]) -> int:
        tot = sum(machines)
        n = len(machines)
        if tot % n:
            return -1
        avg = tot // n
        res = 0
        left_demand = 0
        for num in machines:
            num -= avg
            left_demand += num
            res = max(res, abs(left_demand), num)
        return res
