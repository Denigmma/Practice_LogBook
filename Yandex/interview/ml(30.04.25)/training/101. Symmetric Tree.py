'''
Учитывая root двоичного дерева, проверьте,
является ли оно зеркальным отражением самого себя (т. е. симметричным относительно своего центра).

Пример 1:
Входные данные: root = [1,2,2,3,4,4,3]
Выходные данные: true

Пример 2:
Входные данные: root = [1,2,2,null,3,null,3]
Выходные данные: false

Ограничения:
Количество узлов в дереве находится в диапазоне [1, 1000].
-100 <= Node.val <= 100
'''
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

### Рекурсия O(n) - время, O(n) - память
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def is_mirror(left, right) -> bool:
            if not left and not right:
                return True
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            return is_mirror(left.left, right.right) and is_mirror(left.right, right.left)
        return is_mirror(root.left, root.right) if root else True



### Итеративно O(n) - время, O(n) - память
# class Solution:
#     def isSymmetric(self, root: TreeNode) -> bool:
#         if not root:
#             return True
#
#         stack = [(root.left, root.right)]
#
#         while stack:
#             left, right = stack.pop()
#
#             if not left and not right:
#                 continue
#             if not left or not right:
#                 return False
#             if left.val != right.val:
#                 return False
#
#             # Проверяем зеркальные поддеревья
#             stack.append((left.left, right.right))
#             stack.append((left.right, right.left))
#
#         return True

solution=Solution()
print(solution.isSymmetric([1,2,2,3,4,4,3]))