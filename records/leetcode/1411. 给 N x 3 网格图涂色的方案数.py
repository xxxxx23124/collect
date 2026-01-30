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