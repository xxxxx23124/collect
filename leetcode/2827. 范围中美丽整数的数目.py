# Author: https://leetcode.cn/u/233-zu/

from functools import cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def calc(limit_str: str) -> int:
            n = len(limit_str)

            # 核心，记忆重复出现的状态
            @cache
            def dfs(i: int, is_limit: bool, is_num: bool, rem: int, diff: int) -> int:
                if i == n:
                    return 1 if is_num and rem == 0 and diff == 0 else 0

                res = 0
                upper = int(limit_str[i]) if is_limit else 9
                for d in range(upper + 1):
                    if not is_num:
                        if d == 0:
                            res += dfs(i + 1, is_limit and d == upper, False, rem, diff)
                        else:
                            new_rem = (rem * 10 + d) % k
                            new_diff = diff + (1 if d % 2 == 1 else -1)
                            res += dfs(
                                i + 1, is_limit and d == upper, True, new_rem, new_diff
                            )
                    else:
                        new_rem = (rem * 10 + d) % k
                        new_diff = diff + (1 if d % 2 == 1 else -1)
                        res += dfs(
                            i + 1, is_limit and d == upper, True, new_rem, new_diff
                        )
                return res

            return dfs(0, True, False, 0, 0)

        return calc(str(high)) - calc(str(low - 1))
