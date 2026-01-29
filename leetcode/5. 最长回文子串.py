# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s

        dp = [[False] * n for _ in range(n)]

        max_length = 0
        begin = 0

        for L in range(1, n + 1):
            for i in range(n):
                j = i + L - 1
                if j >= n:
                    break

                if L == 1:
                    dp[i][i] = True
                elif L == 2:
                    if s[i] == s[j]:
                        dp[i][j] = True
                else:
                    if s[i] == s[j] and dp[i + 1][j - 1]:
                        dp[i][j] = True

                if dp[i][j]:
                    if L > max_length:
                        max_length = L
                        begin = i

        return s[begin:begin + max_length]