# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def closestCost(
        self, baseCosts: List[int], toppingCosts: List[int], target: int
    ) -> int:
        baseCosts_len = len(baseCosts)
        search_more = max(baseCosts) + 2 * max(toppingCosts)
        doubleToppingCosts = toppingCosts + toppingCosts
        doubleToppingCosts_len = len(doubleToppingCosts)
        dp = [
            [
                [False] * (doubleToppingCosts_len + 1)
                for _ in range(target + search_more + 1)
            ]
            for __ in range(baseCosts_len)
        ]
        for i in range(baseCosts_len):
            if baseCosts[i] <= target + search_more:
                dp[i][baseCosts[i]][0] = True
        for i in range(baseCosts_len):
            for j in range(1, target + search_more + 1):
                for k in range(1, doubleToppingCosts_len + 1):
                    if doubleToppingCosts[k - 1] <= j:
                        dp[i][j][k] = (
                            dp[i][j][k - 1]
                            or dp[i][j - doubleToppingCosts[k - 1]][k - 1]
                        )
                    else:
                        dp[i][j][k] = dp[i][j][k - 1]
        res = float("inf")
        for i in range(baseCosts_len):
            for j in range(target + search_more, -1, -1):
                if dp[i][j][-1]:
                    if abs(j - target) < abs(res - target):
                        res = j
                    elif abs(j - target) == abs(res - target) and j < res:
                        res = j
        return res