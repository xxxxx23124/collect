# Author: https://leetcode.cn/u/233-zu/

from collections import Counter


class Solution:
    def longestPalindrome(self, s: str) -> int:
        counter = Counter(s)
        is_odd = False
        res = 0
        for count in counter.values():
            if count == 1:
                is_odd = True
            elif count % 2:
                is_odd = True
                count -= 1
                res += count
            else:
                res += count
        if is_odd:
            res += 1
        return res
