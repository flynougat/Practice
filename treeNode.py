import collections
import sys



####################### Binary Search Tree, BST ###########################


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
    

# 86 · Binary Search Tree Iterator
# Next() returns the next smallest element in the BST.
# next() and hasNext() queries run in O(1) time in average.
# Extra memory usage O(h), h is the height of the tree.
# Super Star: Extra memory usage O(1)
#
# use stack
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        while root != None:
            self.stack.append(root)
            root = root.left

    def hasNext(self):
        return len(self.stack) > 0

    def _next(self):
        node = self.stack[-1]
        if node.right is not None:
            n = node.right
            while n != None:
                self.stack.append(n)
                n = n.left
        else:
            n = self.stack.pop()
            while self.stack and self.stack[-1].right == n:
                n = self.stack.pop()
        return node




# 1311 · Lowest Common Ancestor of a Binary Search Tree
# use the characteristics of BST,recursively search

class Solution1:
    def lowestCommonAncestorBST(self, root, p, q):
        if not root:
            return
        # right
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestorBST(root.right, p, q)
        # left
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestorBST(root.left, p, q)
        return root




# 910 · Largest BST Subtree
# subtree contains max number of nodes
# use a helper, divide and conquer
# import sys
    def largestBSTSubtree(self, root):
        if not root:
            return 0
        _, size, _, _ = self.largestBSTSubtreeHelper(root)
        return size

    def largestBSTSubtreeHelper(self, root):
        if not root:
            return True, 0, sys.maxsize, -sys.maxsize
        
        l_bst, l_size, l_min, l_max = self.largestBSTSubtreeHelper(root.left)
        r_bst, r_size, r_min, r_max = self. largestBSTSubtreeHelper(root.right)

        bst = l_bst and r_bst and root.val > l_max and root.val < r_min

        if bst:
            size = l_size + r_size + 1
        else:
            size = max(l_size, r_size)
        
        return bst, size, min(l_min, r_min, root.val), max(l_max, r_max, root.val)



    # 1704 · Range Sum of BST
    def rangeSumBST(self, root, L, R):
        if not root: return 0

        if root.val < L:
            return self.rangeSumBST(root.right, L, R)
        elif root.val > R:
            return self.rangeSumBST(root.left, L, R)
        else:
            return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)



####################### Binary Tree ###########################

# 760 · Binary Tree Right Side View
# Given a binary tree, imagine yourself standing on the right side of it, 
# return the values of the nodes you can see ordered from top to bottom
#
# only return right side
# BFS, 每次queue里最后一个节点就是最右边的.
class Solution:
    def rightSideView(self, root):
        if not root:
            return []

        result = []
        queue = collections.deque([root])

        while queue:
            n = len(queue)
            for i in range(n):
                node = queue.popleft()
                if i == n - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
        return result



# 88 · Lowest Common Ancestor of a Binary Tree
# use divide & conquer

    def lowestCommonAncestorBT(self, root, A, B):
        if not root:
            return 
        
        if root == A or root == B:
            return root
        
        # divide
        left = self.lowestCommonAncestorBT(root.left, A, B)
        right = self.lowestCommonAncestorBT(root.right, A, B)

        # conquer
        # A, B each side
        if left and right:
            return root
        
        # left side
        if left:
            return left
        
        # right side
        if right:
            return right

        # not at left or right 
        return None


# 468 · Symmetric Binary Tree
# recursion

    def isSymmetric(self, root):
        if not root:
            return True
        return self.isSymmetricHelper(root.left, root.right)

    def isSymmetricHelper(self, p, q):
        if p == None and q == None:
            return True
        if p and q and p.val == q.val:
            return self.isSymmetricHelper(p.right, q.left) and self.isSymmetricHelper(p.left, q.right)
        return False


# 7 · Serialize and Deserialize Binary Tree
# use bfs to traverse each level
# record node val in 
    def serialize(self, root):
        if not root:
            return '{}'
        
        queue = collections.deque()
        queue.append(root)
        result = ''
        while queue:
            node = queue.popleft()
            if node:
                result += str(node.val) + ','
                queue.append(node.left)
                queue.append(node.right)
            else:
                result += '#,'
        result = "{" + result[:-1] + "}"
        return result


    def deserialize(self, data):
        if not data or data == '{}':
            return None

        items = data[1:-1].split(',')
        index = 0
        root = TreeNode(items[index])

        queue = collections.deque()
        queue.append(root)
        index += 1

        while index < len(items):
            node = queue.popleft()
            # left first
            if items[index] != '#':
                node.left = TreeNode(items[index])
                queue.append(node.left)
                index += 1
            
            # right
            if items[index] != '#':
                node.right = TreeNode(items[index])
                queue.append(node.right)
                index += 1

        return root
            
# 1115 · Average of Levels in Binary Tree
    def averageOfLevels(self, root):
        if not root:
            return []

        result = []
        queue = collections.deque([root])

        while queue:
            n = len(queue)
            sum_level = 0
            for i in range(n):
                node = queue.popleft()
                sum_level += node.val
                if i == n - 1:
                    result.append(sum_level / n)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
        return result


# 1506 · All Nodes Distance K in Binary Tree 
# Return a list of the values of all nodes that have a distance K from the target node. 
# The answer can be returned in any order.
# Step 1: tree to graph, find neighbors
# Step 2: BFS find k distance neighbors
    def treeToGraph(self, root):
        adjList = {}
        queue = collections.deque()
        queue.append(root)

        while queue:
            curr = queue.popleft()
            if curr not in adjList:
                adjList[curr] = []
            if curr.left:
                adjList[curr].append(curr.left)
                if curr.left not in adjList:
                    adjList[curr.left] = []
                adjList[curr.left].append(curr)
                queue.append(curr.left)
            if curr.right:
                adjList[curr].append(curr.right)
                if curr.right not in adjList:
                    adjList[curr.right] = []
                adjList[curr.right].append(curr)
                queue.append(curr.right)
        return adjList

    def distanceK(self, root, target, K):
        if not root or not target:
            return []
        elif K == 0:
            return [target.val]
        
        adjList = self.treeToGraph(root)

        #BFS
        result = []
        depth = 0
        visited = set()
        queue = collections.deque()
        queue.append(target)

        while queue and depth <= K:
            for _ in range(len(queue)):
                curr = queue.popleft()
                if curr not in visited:
                    for child in adjList[curr]:
                        queue.append(child)
                    if depth == K:
                        result.append(curr.val)
                visited.add(curr)
            depth += 1

        return result