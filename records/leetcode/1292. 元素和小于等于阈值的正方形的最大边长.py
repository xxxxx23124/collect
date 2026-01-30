# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        m, n = len(mat), len(mat[0])
        P = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                P[i][j] = P[i-1][j] + P[i][j-1] - P[i-1][j-1] + mat[i-1][j-1]
        ans = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                target_len = ans + 1
                if i >= target_len and j >= target_len:
                    r1, c1 = i - target_len, j - target_len
                    current_sum = P[i][j] - P[r1][j] - P[i][c1] + P[r1][c1]
                    if current_sum <= threshold:
                        ans += 1
        return ans