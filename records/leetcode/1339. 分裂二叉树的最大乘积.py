# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        sub_tree_sums = []

        def post_order(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            sub_tree_sum = post_order(node.left) + post_order(node.right) + node.val
            sub_tree_sums.append(sub_tree_sum)
            return sub_tree_sum
        total = post_order(root)
        ans = 0
        for sub_tree_sum in sub_tree_sums:
            ans = max(ans, sub_tree_sum*(total-sub_tree_sum))
        return ans % (10**9 + 7)