# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        MOD = 1337
        res = 1
        for e in reversed(b):
            res = (res * pow(a, e, MOD)) % MOD
            a = pow(a, 10, MOD)
        return res
