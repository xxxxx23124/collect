# Author: https://leetcode.cn/u/233-zu/

from typing import List
from collections import Counter


class Solution:
    MAX_N = 10**5
    # get Smallest Prime Factor table
    _temp_spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if _temp_spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if _temp_spf[j] == j:
                    _temp_spf[j] = i
    # tuple for safe
    spf = tuple(_temp_spf)
    del _temp_spf

    def sumFourDivisors(self, nums: List[int]) -> int:
        counts = Counter(nums)
        keys = sorted(counts.keys())
        ans = 0
        for i in keys:
            if self.__count_divisors_with_spf(i) == 4:
                ans += counts[i] * (1 + i + (i // self.spf[i] + self.spf[i]))
        return ans

    # 其实不需要算出确切的因数个数，不过为了普适就这样写了
    def __count_divisors_with_spf(self, n) -> int:
        if n == 1:
            return 1
        total_divisors = 1
        temp = n
        while temp > 1:
            p = self.spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            total_divisors *= count + 1
        return total_divisors
