# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        possible_sums = {0}

        for row in mat:
            next_sums = set()
            for num in row:
                for s in possible_sums:
                    next_sums.add(s + num)
            possible_sums = next_sums

        min_diff = float('inf')
        for s in possible_sums:
            diff = abs(s - target)
            if diff < min_diff:
                min_diff = diff

        return min_diff