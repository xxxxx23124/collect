# Author: https://leetcode.cn/u/233-zu/

class Solution:
    def subtreeWithAllDeepest(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode]):
            if node is None:
                return 0, None
            left_depth, left_lca = dfs(node.left)
            right_depth, right_lca = dfs(node.right)
            if left_depth > right_depth:
                return left_depth + 1, left_lca
            elif left_depth < right_depth:
                return right_depth + 1, right_lca
            else:
                return left_depth + 1, node

        return dfs(root)[1]