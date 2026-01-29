# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        s_len = len(s)
        if s_len == 0:
            return 0

        dp = [0] * s_len
        for i in range(1, s_len):
            if s[i] == ")":
                if s[i - 1] == "(":
                    if i >= 2:
                        dp[i] = dp[i - 2] + 2
                    else:
                        dp[i] = 2
                elif (
                    s[i - 1] == ")"
                    and i - dp[i - 1] - 1 >= 0
                    and s[i - dp[i - 1] - 1] == "("
                ):
                    if i - dp[i - 1] - 2 >= 0:
                        dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
                    else:
                        dp[i] = dp[i - 1] + 2

        return max(dp)
