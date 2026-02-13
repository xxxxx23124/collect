# Author: https://leetcode.cn/u/233-zu/

class SolutionA:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        nums_len = len(nums)
        if nums_len == 1:
            return 1
        res = []
        for i in range(nums_len - 1):
            diff = nums[i + 1] - nums[i]
            if diff == 0:
                continue
            elif len(res) and diff * res[-1] < 0:
                res.append(diff)
            elif len(res) == 0:
                res.append(diff)
        if len(res) and abs(res[-1]) > 0:
            res.append(nums[-1])
        elif len(res) == 0:
            res.append(nums[-1])

        return len(res)

class SolutionB:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        up = down = 1
        nums_len = len(nums)
        for i in range(nums_len - 1):
            if nums[i] < nums[i + 1]:
                up = down + 1
            elif nums[i] > nums[i + 1]:
                down = up + 1
        return max(up, down)