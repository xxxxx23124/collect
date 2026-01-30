# Author: https://leetcode.cn/u/233-zu/

class SolutionA:
    def canJump(self, nums: List[int]) -> bool:
        nums_len = len(nums)
        dp = [False] * nums_len
        dp[0] = True
        for i in range(nums_len):
            if dp[i]:
                for j in range(nums[i]):
                    if i + j + 1 < nums_len:
                        dp[i + j + 1] = True
                    else:
                        dp[-1] = True
                        return True
        return dp[-1]

class SolutionB:
    def canJump(self, nums: List[int]) -> bool:
        nums_len = len(nums)
        can_jump = 0
        for i in range(nums_len):
            if can_jump - i >= 0:
                can_jump = max(can_jump, i + nums[i])
                if can_jump >= nums_len - 1:
                    return True
            else:
                break
        return False