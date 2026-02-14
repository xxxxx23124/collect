# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        num_len = len(num)
        stk = []
        for i in range(num_len):
            while k and len(stk) > 0 and stk[-1] > num[i]:
                stk.pop()
                k -= 1
            stk.append(num[i])

        stk = stk[:-k] if k else stk

        return "".join(stk).lstrip('0') or "0"
