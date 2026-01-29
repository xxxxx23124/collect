# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def trap(self, height: List[int]) -> int:
        h_len = len(height)
        if h_len == 0:
            return 0
        left_max = [0] * h_len
        rigth_max = [0] * h_len
        ans_min = [0] * h_len

        max_h = 0
        for i in range(h_len):
            cur_h = height[i]
            max_h = max(cur_h, max_h)
            left_max[i] = max_h

        max_h = 0
        for i in range(h_len - 1, -1, -1):
            cur_h = height[i]
            max_h = max(cur_h, max_h)
            rigth_max[i] = max_h

        for i in range(h_len):
            ans_min[i] = min(left_max[i], rigth_max[i])

        ans = 0
        for i in range(h_len):
            ans += (ans_min[i] - height[i])

        return ans