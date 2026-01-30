# Author: https://leetcode.cn/u/233-zu/

from functools import cache

class Solution:
    @cache
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return ['']
        ans = []
        for i in range(n):
            # (a)b
            for a in self.generateParenthesis(i):
                for b in self.generateParenthesis(n - 1 - i):
                    ans.append(f'({a}){b}')
        return ans