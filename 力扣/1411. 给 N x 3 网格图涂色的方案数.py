
"""
1411. 给 N x 3 网格图涂色的方案数
你有一个 n x 3 的网格图 grid ，你需要用 红，黄，绿 三种颜色之一给每一个格子上色，且确保相邻格子颜色不同（也就是有相同水平边或者垂直边的格子颜色不同）。

给你网格图的行数 n 。

请你返回给 grid 涂色的方案数。由于答案可能会非常大，请你返回答案对 10^9 + 7 取余的结果。
"""
# Author: https://leetcode.cn/u/233-zu/
class Solution:
    def numOfWays(self, n: int) -> int:
        mod = 10**9 + 7

        # 0: red 1: yellow 2: green
        valid_1d = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and j != k:
                        valid_1d.append((i, j, k))
        valid_1d_num = len(valid_1d)

        # valid pattern between two part
        # i->top j->bottom
        valid_pattern = [[] for _ in range(valid_1d_num)]
        for i in range(valid_1d_num):
            top_left, top_mid, top_right = valid_1d[i]
            for j in range(valid_1d_num):
                bottom_left, bottom_mid, bottom_right = valid_1d[j]
                if (
                    top_left != bottom_left
                    and top_mid != bottom_mid
                    and top_right != bottom_right
                ):
                    valid_pattern[i].append(j)

        # N(top pattern) + 1(bottom pattern)
        # j -> new bottom, i -> old bottom
        dp = [1] * valid_1d_num
        for _ in range(n - 1):
            new_dp = [0] * valid_1d_num
            for i in range(valid_1d_num):
                current_count = dp[i]
                for j in valid_pattern[i]:
                    new_dp[j] = (new_dp[j] + current_count) % mod
            dp = new_dp

        return sum(dp) % mod