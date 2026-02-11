# Author: https://leetcode.cn/u/233-zu/

from collections import Counter


class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        counter = Counter(s)
        stk = []

        for c in s:
            if c not in stk:
                while len(stk) and stk[-1] > c:
                    if counter[stk[-1]] > 0:
                        stk.pop()
                    else:
                        break
                stk.append(c)
            counter[c] -= 1

        return "".join(stk)
