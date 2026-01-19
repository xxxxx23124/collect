"""
2827. 范围中美丽整数的数目
给你正整数 low ，high 和 k 。

如果一个数满足以下两个条件，那么它是 美丽的 ：

偶数数位的数目与奇数数位的数目相同。
这个整数可以被 k 整除。
请你返回范围 [low, high] 中美丽整数的数目。



示例 1：

输入：low = 10, high = 20, k = 3
输出：2
解释：给定范围中有 2 个美丽数字：[12,18]
- 12 是美丽整数，因为它有 1 个奇数数位和 1 个偶数数位，而且可以被 k = 3 整除。
- 18 是美丽整数，因为它有 1 个奇数数位和 1 个偶数数位，而且可以被 k = 3 整除。
以下是一些不是美丽整数的例子：
- 16 不是美丽整数，因为它不能被 k = 3 整除。
- 15 不是美丽整数，因为它的奇数数位和偶数数位的数目不相等。
给定范围内总共有 2 个美丽整数。
示例 2：

输入：low = 1, high = 10, k = 1
输出：1
解释：给定范围中有 1 个美丽数字：[10]
- 10 是美丽整数，因为它有 1 个奇数数位和 1 个偶数数位，而且可以被 k = 1 整除。
给定范围内总共有 1 个美丽整数。
示例 3：

输入：low = 5, high = 5, k = 2
输出：0
解释：给定范围中有 0 个美丽数字。
- 5 不是美丽整数，因为它的奇数数位和偶数数位的数目不相等。


提示：

0 < low <= high <= 10^9
0 < k <= 20
"""

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
