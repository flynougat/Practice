import collections


class Solution:


####################### Binary Search ###########################

    # 14 · First Position of Target
    def binarySearch(self, nums, target):
        if not nums or not target: 
            return -1
        
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2
            if nums[mid] < target:
                start = mid
            else:
                end = mid

        if nums[end] == target:
            return end
        if nums[start] == target:
            return start

        return -1


    # 460 · Find K Closest Elements
    # binary search find the target first, then use pointer to travel left, right
    def kClosestNumbers(self, A, target, k):
        if not A:
            return []

        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if A[mid] < target:
                start = mid
            else:
                end = mid
        
        left = start
        right = end
        result = []
        while k > 0:
            if left >= 0 and right < len(A):
                if target - A[left] <= A[right] - target:
                    result.append(A[left])
                    left -= 1
                    k -= 1
                else:
                    result.append(A[right])
                    right += 1
                    k -= 1
            elif left < 0 and right < len(A): #left side is shorter, still can find right
                result.append(A[right])
                right += 1
                k -= 1
            else: # right side short, still can find left
                result.append(A[left])
                left -= 1
                k -= 1

        return result


    # 437 · Copy Books
    # binary search complete time
    # lower bound: max(pages), 理解为一共有len(pages)个人，每个人只复印一本书 
    # upper bound: sum(pages), 理解为只有一个人要独自复印所有的书
    # use a greedy function to check how many people needed. if value <= k menas more
    # labor available, time could be less. if value > k means not enough labor
    def copyBooks(self, pages, k):
        if not pages:
            return 0

        start = max(pages)
        end = sum(pages)

        while start + 1 < end:
            mid = (start + end) // 2
            if self.can_complete(pages, k, mid):
                end = mid
            else:
                start = mid

        if self.can_complete(pages, k, start):
            return start
        
        return end

    def can_complete(self, pages, k, goal):
        worker = 1
        pageSum = 0
        for page in pages:
            if pageSum + page <= goal:
                pageSum += page
            else:
                worker += 1
                pageSum = page

        return worker <= k


####################### PrefixSum + HashMap ###########################

    # 1712 · Binary Subarrays With Sum
    # prefixSum + HashMap
    # sum[i:j] -> prefix[j] - prefix[i-1]
    # 固定j，找i  X X X [i X X j] X X
    # hashmap to store key:prefixSum value: count of idx
    def numSubarraysWithSum(self, A, S):
        if not A:
            return 0

        counter = collections.defaultdict(int)
        counter[0] = 1  # must use 1 because imaging an empty subarry in front`

        prefixSum, result = 0, 0

        for i in A:
            prefixSum += i
            result += counter[prefixSum - S]
            counter[prefixSum] += 1

        return result


    # 994 · Contiguous Array
    # [0,1,1,0,0] find max length of equal 0 and 1
    # change 0 to -1, use prefixSum + HashMap
    # hashmap key: prefixSum, value: index
    def findMaxLength(self, nums):
        prefixSum = [0] * (len(nums) + 1)
        for i in range(1, len(prefixSum)):
            prefixSum[i] = prefixSum[i-1] + (1 if nums[i-1] == 1 else -1)
        map = {0 : 0}
        max_length = 0
        for i in range(len(nums)):
            if prefixSum[i+1] in map:
                start_index = map[prefixSum[i+1]] - 1
                curr_length = i - start_index
                max_length = max(max_length, curr_length)
            if prefixSum[i+1] not in map:
                map[prefixSum[i+1]] = i + 1
        return max_length
        




####################### Two Pointers ###########################


    # 627 · Longest Palindrome
    # traverse string, use map, if char exists in map, delete and add 2 length,
    # if not, add to map. if map remains, length add 1 
    def longestPalindrome(self, s):
        if not s:
            return 0
        length = 0
        charMap = {}

        for char in s:
            if char not in charMap:
                charMap[char] = 1
            else:
                charMap.pop(char)
                length += 2

        if len(charMap) > 0:
                length += 1

        return length


    # 64 · Merge Sorted Array
    # Two pointers -- Compare from right to left, mth of A and nth of B, 
    # put the larger one to A[m+n-1], then move the pointers.
    def mergeSortedArray(self, A, m, B, n):
        if A is None or B is None:
            return

        i = m - 1
        j = n - 1
        k = m + n - 1

        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[k] = A[i]
                i -= 1
            else:
                A[k] = B[j]
                j -= 1
            k -= 1
        
        while j >= 0:
            A[k] = B[j]
            j -= 1
            k -= 1



    # 382 · Triangle Count
    # two shorter lines sum > longest
    # method 1: brute force O(n^3)
    # method 2: sort, use 2 for loops to lock shorter lines and binary search longer one. O(n^2logn)
    # methos 3: two pointer. Lock longest first i, then use two pointer to find 2 shorter ones between [0, i -1]
    def triangleCount(self, S):
        result = 0
        S.sort()

        for i in range(len(S) - 1, -1, -1):
            left, right = 0, i - 1
            while left < right:
                if S[left] + S[right] > S[i]:
    # all posibilities between left and right are satisfactory because we can increase left right - left times
                    result += right - left 
                    right -= 1
                else:
                    left += 1

        return result



####################### Sort ###########################

    # 463 · Sort Integers
    def sortIntegers(self, A):
        if not A:
            return

        for i in range(1, len(A)):
            for j in range(i-1, -1, -1):
                if A[i] < A[j]:
                    A[i], A[j] = A[j], A[i]
                    i = j
                else:
                    break
        return A


    # 464 · Sort Integers II
    # use Quick sort
    def sortIntegers2(self, A):
        self.quickSort(A, 0, len(A) - 1)

    def quickSort(self, A, start, end):
        if start >= end:
            return

        left, right = start, end
        pivot = A[(start + end) // 2]

        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot:
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
            
        self.quickSort(A, start, right)
        self.quickSort(A, left, end)


####################### Union Find ###########################
# 1. Find whether 2 elements in the same group (find)
# 2. Merge 2 groups of elements (union)
class UnionFind:
    def __init__(self):
        self.father = {}
        self.set_size = {}
        self.num_of_set = 0
    
    def add(self, x):
        # if already added, just return
        if x in self.father:
            return
        self.father[x] = None
        self.set_size[x] = 1
        self.num_of_set += 1

    def find(self, x):
        # pointer root points to x, keep finding root's father until root point to x's root
        root = x
        while self.father[root] != None:
            root = self.father[root]
        
        while x != root:
            original_father = self.father[x] # temprorily store x's father
            self.father[x] = root  # point x to root
            x = original_father  # pointer moves up to x's father

        return root

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)  # find root

        if root_x != root_y:
            self.father[root_x] = root_y   # 将一个点的根变成新的根
            self.num_of_set -= 1 # 集合数量减少1
            self.set_size[root_y] += self.set_size[root_x] # 计算新的根所在集合大小

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)

    def get_num_of_set(self):
        return self.num_of_set

    def get_size_of_set(self, x):
        return self.size_of_set[self.find(x)]


####################### DFS ###########################

    # 1857 · Find Friend Circle Number
    def findCircleNum(self, M):
        n = len(M)
        count = 0
        self.visit = [0] * n

        for i in range(n):
            if self.visit[i] == 0:
                count += 1
                self.findCircleNumDFS(i, M)

        return count

    def __init__(self):
        self.visit = []

    def findCircleNumDFS(self, index, M):
        for i in range(len(M)):
            if M[index][i] == 1 and self.visit[i] == 0:
                self.visit[i] = 1
                self.findCircleNumDFS(i, M)



    # 1891 · Travel Plan
    def travelPlan(self, arr):
        result = []
        visited = [False] * len(arr)
        visited[0] = True
        self.travelPlanDFS(arr, visited, 0, 1, 0, result)
        return min(result)
    def travelPlanDFS(self, graph, visited, last_city, num_visited, total_cost, result):
        if num_visited == len(graph):
            result.append(total_cost + graph[last_city][0])
        
        for i in range(1, len(graph)):
            if visited[i]:
                continue
            visited[i] = True
            self.travelPlanDFS(graph, visited, i, num_visited + 1, total_cost + graph[last_city][i], result)
            visited[i] = False




    # 1811 · Find Maximum Gold
    def FindMaximumGold(self, grids):
        result = 0
        visited = [[False] * len(grids[0]) for _ in range(len(grids))]
        for i in range(len(grids)):
            for j in range(len(grids[0])):
                visited[i][j] = True
                result = max(result, self.FindMaximumGoldDFS(grids, visited, (i, j), grids[i][j]))
                visited[i][j] = False
        return result
    def FindMaximumGoldDFS(self, grids, visited, curr, gold):
        max_gold = gold
        DIR = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        x, y = curr
        for dx, dy in DIR:
            next_x, next_y = x + dx, y + dy
            new_pos = (next_x, next_y)
            if not 0 <= next_x < len(grids) or not 0 <= next_y < len(grids[0]):
                continue
            if grids[next_x][next_y] == 0:
                continue
            if visited[next_x][next_y]:
                continue
            visited[next_x][next_y] = True
            max_gold = max(max_gold, self.FindMaximumGoldDFS(grids, visited, new_pos, gold + grids[next_x][next_y]))
            visited[next_x][next_y] = False
        return max_gold


    # 990 · Beautiful Arrangement
    def countArrangement(self, N):
        visited = [False] * N
        results = []
        self.countArrangementDFS(visited, [], results)
        return len(results)

    def countArrangementDFS(self, visited, result, results):
        if len(result) == len(visited):
            results.append(list(result))

        for i in range(len(visited)):
            if visited[i]:
                continue
                
            index = len(result)
            if (index + 1) % (i + 1) != 0 and (i + 1) % (index + 1) != 0:
                continue

            result.append(i)
            visited[i] = True
            self.countArrangementDFS(visited, result, results)
            visited[i] = False
            result.pop()




####################### Linked List ###########################
    #Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

    # 102 Linked List Cycle
    # Given a linked list, determine if it has a cycle in it.
    # Challenge: solve it without using extra space
    # use slow and fast pointers
    # if no cycle, faster pointer will reach null, return false
    # if there is cycle, slow and faster pointer will meet eventrually
    def hasCycle(self, head):
        slow, fast = head, head
        while fast and fast.next:
            slow  = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False



    # 103 · Linked List Cycle II
    # Given a linked list, return the node where the cycle begins.
    def detectCycle(self, head):
        # write your code here
        if not head or not head.next:
            return None

        slow, fast = head, head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                break

        if fast and slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
        
        return None



    # 221 · Add Two Numbers II
    def addLists2(self, l1, l2):
        str1 = ''
        str2 = ''

        while l1:
            str1 += str(l1.val)
            l1 = l1.next

        while l2:
            str2 += str(l2.val)
            l2 = l2.next

        sum_num = int(str1) + int(str2)
        sum_str = str(sum_num)

        head = ListNode(int(sum_str[0]))
        temp = head
        for i in range(1, len(sum_str)):
            temp.next = ListNode(int(sum_str[i]))
            temp = temp.next

        return head


    # 450 · Reverse Nodes in k-Group
    def reverseKGroup(self, head, k):
        # write your code here
        # step 1 traverse, check linked list length. if length  < k, return head
        # step 2 recursively - repeat reverse every k interval
        if not head or not head.next or k <=1 :
            return head
        
        temp = head
        count = 0
        while temp and count < k:
            temp = temp.next
            count += 1
        if count < k:
            return head

        pre = head
        cur = head.next
        i = 1
        while i < k:
            nex = cur.next
            cur.next = pre
            pre = cur
            cur = nex
            i += 1
        head.next = self.reverseKGroup(cur, k)

        return pre







####################### BST ###########################

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


    # 915 · Inorder Predecessor in BST
    # Given a binary search tree and a node in it, find the in-order predecessor of that node in the BST.
    def inorderPredecessor(self, root, p):
        if not root:
            return None

        predecessor = None
        while root:
            if p.val <= root.val:
                root = root.left
            else:
                predecessor = root
                root = root.right

        return predecessor 


    # 902 · Kth Smallest Element in a BST
    # In-order traverse tree 
    # put seen node into stack to track parent
    # pop out node from stack when visit, put the node into visted arry, count
    # when count == k, return nodel value
    def kthSmallest(self, root, k):
        visited = []
        stack = collections.deque([])
        node = root
        while node:
            stack.append(node)
            node = node.left
        while stack:
            visitNode = stack.pop()
            visited.append(visitNode.val)
            visitNode = visitNode.right
            stack.append(visitNode)
        return visited[k-1]





####################### DP ###########################

# 1. Split into sub-problem
    # "sub-problem": the same problem as before but in a smaller scope or circumstances
# 2. Reuse the solution you are implemented
# 3. Save intermediate result  ---dp  dynamic programming

