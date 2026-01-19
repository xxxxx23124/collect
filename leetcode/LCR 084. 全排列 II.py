"""
LCR 084. 全排列 II
给定一个可包含重复数字的整数集合 nums ，按任意顺序 返回它所有不重复的全排列。



示例 1：

输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
示例 2：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]


提示：

1 <= nums.length <= 8
-10 <= nums[i] <= 10

"""

# Author: https://leetcode.cn/u/233-zu/

from collections import Counter


class SolutionA:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        counts = Counter(nums)
        n = len(nums)
        res = []

        def backtrack(path):
            if len(path) == n:
                res.append(list(path))
                return

            for num in counts:
                if counts[num] > 0:
                    path.append(num)
                    counts[num] -= 1
                    backtrack(path)
                    counts[num] += 1
                    path.pop()

        backtrack([])
        return res

class SolutionB:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        used = [False] * n
        res = []
        def backtrack(path):
            if len(path) == n:
                res.append(list(path))
                return
            for i in range(n):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                path.append(nums[i])
                used[i] = True
                backtrack(path)
                used[i] = False
                path.pop()
        backtrack([])
        return res