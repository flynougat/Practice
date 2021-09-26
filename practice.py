import collections
from collections import deque
import heapq
from os import system
import sys
from typing import Counter
from typing import (
    List,
)

class Solution:

    ####################### Binary Search (Array) ###########################

    # 160 Find Minimum in Rotated Sorted Array II
    # use Binary Search
    def findMin(self, nums):
        left, right = 0, len(nums) - 1

        while (left < right):
            if nums[left] < nums[right]:
                return nums[left]

            # in Python 3, 6/5 = 1.2, 6//5 = 1
            mid = left + (right - left) // 2
            
            # smallest in left side
            if nums[left] > nums[mid]:
                right = mid
            # smallest in right side
            elif nums[left] < nums[mid]:
                left = mid + 1
            # hard to decide, left + 1
            else:
                left += 1
        return nums[left]




    ####################### Stack ###########################


    # 12 · Min Stack
    # when push to stack, if new < top, new is the new top. if new > top, push top again
    # O(1) time to return smallest in stack
    def __init__(self):
        # do intialization if necessary
        self.stack = []
        self.min_stack = []

    def push(self, number):
        # write your code here
        self.stack.append(number)
        if len(self.min_stack)== 0 or number < self.min_stack[len(self.min_stack) - 1]:
            self.min_stack.append(number)
        else:
            self.min_stack.append(self.min_stack[len(self.min_stack) - 1])

    def pop(self):
        # write your code here
        self.min_stack.pop()
        return self.stack.pop()

    def min(self):
        # only return, not pop. pop will remove element
        return self.min_stack[len(self.min_stack) - 1]

    

    # 980 · Basic Calculator II
    # The expression string contains only non-negative integers, +, -, *, / operators 
    # and empty spaces . 
    # Division of integers should round off decimals.
    # 通过栈来实现运算，按顺序读取字符串，将第一个数入栈。之后遇到+，将下一个数num入栈；
    # 遇到-，则将-num入栈；遇到乘或除，先将上一个数出栈，与当前数进行运算后，再将结果入栈。
    # 读取完整个字符串后，将栈中所有的数相加即运算结果。
    def calculate(self, s):
        if not s:
            return "0"
        
        stack, num, operator = [], 0, '+'

        for i, c in enumerate(s):
            if c.isdigit():
                num = 10 * num + int(c)
            if (not c.isdigit() and not c.isspace()) or i == len(s) - 1:
                if operator == '+':
                    stack.append(num)
                elif operator == '-':
                    stack.append(-num)
                elif operator == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                operator = c
                num = 0
        
        return sum(stack)


    # 363 · Trapping Rain Water 
    # 用单调递减栈，当heights[i] >= heights[stack[-1]], pop 栈直到栈顶大于当前数，压栈，并算盛水量
    # also can use Tow Pointer
    def trapRainWater(self, heights):
        if not heights: return 0

        result = 0
        stack = []

        for i, height in enumerate(heights):
            while stack and height >= heights[stack[-1]]:
                ground_height = heights[stack.pop()]
                if not stack:
                    continue
                low_index = stack[-1]
                water_line = min(heights[low_index], height)
                result += (water_line - ground_height) * (i - low_index - 1)

            stack.append(i)
        
        return result



####################### String ###########################

    #1889 · Interval Merge (String) 
    # return true if string intervals can b merged
    # O(1) time and space
    def MergeJudge(self, interval_A, interval_B):
            # get first and last chars from A, B
            firstChar_A = interval_A.split(interval_A[0])[1].split(',')[0]
            lastChar_A = interval_A.split(',')[1].split(interval_A[-1])[0]
            firstChar_B = interval_B.split(interval_B[0])[1].split(',')[0]
            lastChar_B = interval_B.split(',')[1].split(interval_B[-1])[0]

            if firstChar_A > firstChar_B:
                if lastChar_B > firstChar_A:
                    return True
                if (lastChar_B == firstChar_A) and (interval_A[0] == '[') or (interval_B[-1] == ']'):
                    return True
                if ((lastChar_B + "a") == firstChar_A) and (interval_B[-1] == ']') and (interval_A[0] == '['):
                    return True
                return False
            else:
                if lastChar_A > firstChar_B:
                    return True
                if (lastChar_A == firstChar_B) and (interval_A[-1] == ']') or (interval_B[0] == '['):
                    return True
                if ((lastChar_A + "a") == firstChar_B) and (interval_A[-1] == ']') and (interval_B[0] == '['):
                    return True    
                return False



    # 418 · Integer to Roman
    # Given an integer, convert it to a roman numeral.
    def intToRoman(self, n):
        digit = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        rom = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        result = ''

        for d, r in zip(digit, rom):
            result += r * (n//d)
            n = n % d
        return result


    # 1263 · Is Subsequence
    # Given a string s and a string t, check if s is subsequence of t
    # 直接按s串的顺序去遍历t串
    def isSubsequence(self, s, t):
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        if i == len(s):
            return True
        return False


    # 647 · Find All Anagrams in a String
    # Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
    # If s is an anagram of p, then s is a permutation of p.
    # sliding window
    # time O(n), space O(1)
    def findAnagrams(self, s, p):
        list = []
        times = dict()
        # find how many times each letter occurs in p
        for c in p:
            if c not in times:
                times[c] = 1
            else:
                times[c] += 1
        # create slide window
        l, r = 0, -1
        while l < len(s):
            if r - l + 1 == len(p):
                list.append(l)
            if r + 1 < len(s) and s[r+1] in times and times[s[r+1]] > 0:
                r += 1
                times[s[r]] -= 1
            else:
                if s[l] not in times:
                    times[s[l]] = 1
                else:
                    times[s[l]] += 1
                l += 1
        return list


    # 1394 · Goat Latin
    def  toGoatLatin(self, S):
        vowel = set('aeiouAEIOU')
        word = []
        space = ' '

        for i , w in enumerate(S.split()):
            if w[0] not in vowel:
                word.append(w[1:] + w[0] + 'ma' + 'a'*(i+1))
            else:
                word.append(w + 'ma' + 'a'*(i+1))

        return space.join(word)


    # 1721 · Minimum Add to Make Parentheses Valid
    def minAddToMakeValid(self, S):
        left = right = 0
        for i in S:
            if right == 0 and i == ')':
                left += 1
            else:
                right += 1 if i == '(' else -1
        return left + right



    # 1299 · Bulls and Cows
    def getHint(self, secret, guess):
        secret_count = {str(i): 0 for i in range(0, 10)}
        guess_count = {str(i): 0 for i in range(0, 10)}

        count_A = 0
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                count_A += 1
            else:
                secret_count[secret[i]] += 1
                guess_count[guess[i]] += 1

        count_B = 0
        for num in guess_count:
            count_B += min(secret_count[num], guess_count[num])

        return str(count_A) + 'A' + str(count_B) + 'B'





    # 1876 · Alien Dictionary(easy)
    #def isAlienSorted(self, words, order):

####################### Matrix ###########################

    # 433 · Number of Islands
    def numIslands(self, grid):
        if grid == None or len(grid) == 0:
            return 0

        result = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    self.islandBFS(grid, i, j)
                    result += 1

        return result

    def islandBFS(self, grid, x, y):
        check = [(0,1), (0,-1), (1,0), (-1,0)]
        queue = deque([(x, y)])
        
        while queue:
            x, y = queue.pop()
            for dx, dy in check:
                if 0 <= dx + x < len(grid) and 0 <= dy + y < len(grid[0]) and grid[dx + x][dy + y] == 1:
                    grid[dx + x][dy + y] = 0
                    queue.append((dx + x, dy + y))




    # 1272 · Kth Smallest Element in a Sorted Matrix
    # Given a n x n matrix where each of the rows and columns are sorted in ascending order, 
    # find the kth smallest element in the matrix.
    # Matrix + binary search
    def kthSmallest(self, matrix, k):
        m = len(matrix)
        n = len(matrix[0])
        low, high = matrix[0][0], matrix[m-1][n-1]
        while low + 1 < high:
            mid = low + (high - low) // 2
            count = self.kthSmallestHelper(matrix, mid)
            if count < k:
                low = mid
            else: high = mid
        if self.kthSmallestHelper(matrix, low) >= k:
            return low
        return high

    def kthSmallestHelper(self, matrix, target):
        m = len(matrix)
        n = len(matrix[0])
        i, j = m - 1, 0
        count = 0
        while i >=0 and j < n:
            if matrix[i][j] <= target:
                count += i +1
                j += 1
            else:
                i -= 1
        return count



    # 1205 · Diagonal Traverse
    # segments = m + n -1
    # each element on diagonal has the same i+j value. 
    # i+j odd segment, down left order i++, j--
    # i+j even segment up right order i-- j++ (reverse)
    # j organized the order
    def findDiagonalOrder(self, matrix):
        if not matrix or not matrix[0]: return []

        m, n = len(matrix), len(matrix[0])
        segments = [[] for _ in range(m+n-1)]

        for i in range(m):
            for j in range(n):
                segments[i+j].append(matrix[i][j])

        result = []

        for x in range(len(segments)):
            if x % 2 == 1:
                result.extend(segments[x])
            else:
                result.extend(segments[x][::-1])
        
        return result


    # 1042 · Toeplitz Matrix 
    # A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.
    def isToeplitzMatrix(self, matrix):
        m = len(matrix)
        n = len(matrix[0])

        if m == 1 and n == 1: return True

        for i in range(m - 1):
            for j in range(n - 1):
                if matrix[i][j] != matrix[i+1][j+1]:
                    return False
        return True



####################### Topological Sort ###########################
    # 拓扑排序步骤：
    # 1.建图并记录所有节点的入度。
    # 2.将所有入度为0的节点加入队列。
    # 3.取出队首的元素now，将其加入拓扑序列。
    # 4.访问所有now的邻接点nxt，将nxt的入度减1，当减到0后，将nxt加入队列。
    # 5.重复步骤3、4，直到队列为空。
    # 6.如果拓扑序列个数等于节点数，代表该有向图无环，且存在拓扑序。
    # Time and Space O(V+E)

    # 615 · Course Schedule
    def canFinish(self, numCourses, prerequisites):




    # 616 · Course Schedule II

    def findOrder(self, numCourses, prerequisites):
        # count indegrees
        edges = {i : [] for i in range(numCourses)}
        degrees = [0 for i in range(numCourses)]
        for i, j in prerequisites:
            edges[j].append(i)
            degrees[i] += 1
        
        # push 0 indegree to queue
        queue = []
        result = []
        for i in range(numCourses):
            if degrees[i] == 0:
                queue.append(i)

        # bfs
        while queue:
            course = queue.pop(0)
            result.append(course)
            for e in edges[course]:
                degrees[e] -= 1
                if degrees[e] == 0:
                    queue.append(e)
        if len(result) == numCourses:
            return result
        else:
            return []


####################### Array ###########################

    # 606 · Kth Largest Element II
    #  N is much larger than k
    # 三种解法：
    # QuickSelect：平均O(n)，最坏O(n ^ 2)
    # PriorityQueue：O(nlogk)
    # Heapify：O(n + klogn)
    # Because n >> k, heap will be the best solution

    def kthLargestElement2(self, nums, k):
        if not nums or k < 1:
            return None
        # import heapq
        # 先造一个k - 1数量的minHeap， 再从nums里面k至结尾来update minHeap
        min_heap = nums[:k]
        heapq.heapify(min_heap)

        for num in nums[k:]:
            heapq.heappushpop(min_heap, num)
        
        return min_heap[0]



    # 362 · Sliding Window Maximum
    # in the While loop
    # 1. kick out the number smaller than current from queue
    # 2. put index of current number into queue
    # 3. start output when there are enough numbers in queue
    # 4. remove old numbers from front of queue
    def maxSlidingWindow(self, nums, k):

        queue, result = deque([]), []

        for i in range(len(nums)):
            while queue and nums[i] >= nums[queue[-1]]:
                queue.pop()
            queue.append(i)

            # add num with index queue[0] to result
            if i + 1 >= k:
                result.append(nums[queue[0]])
            # remove index queue[0] from queue
            if i + 1 - k == queue[0]:
                queue.popleft()
        return result



    # 1212 · Max Consecutive Ones
    # Given a binary array, find the maximum number of consecutive 1s in this array.
    def findMaxConsecutiveOnes(self, nums):
        result = 0
        count = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                count += 1
                result = max(count, result)
            else:
                count = 0
        return result


    # 1310 · Product of Array Except Self
    # Brute force O(n^2)
    # 令result为一个全是1的数组。令前缀积prefixProduct等于1，后缀积postfixProduct等于1。
    # 正序遍历数组nums，第i个位置先将结果乘上prefixProduct，再将prefixProduct乘上nums[i]。
    # 逆序遍历数组nums，第i个位置先将结果乘上postfixProduct，再将postfixProduct乘上nums[i]。
    # 返回result。遍历两次数组，时间复杂度为O(N)。只需额外O(1)的时间，记录前缀积和后缀积
    def productExceptSelf(self, nums):
        length = len(nums)
        result = [1] * length # create length size array and put 1 for each element
        prefix_product = 1
        postfix_product = 1

        for i in range(length):
            result[i] *= prefix_product
            prefix_product *= nums[i]

        for i in range(length - 1, -1, -1):  # range(start, stop, step)
            result[i] *= postfix_product
            postfix_product *= nums[i]

        return result


    # 1183 · Single Element in a Sorted Array
    # Input: [1,1,2,3,3,4,4,8,8] Output: 2
    # use binary search. if nums[mid] != nums[mid + 1], the single is in the left half
    def singleNonDuplicate(self, nums):
        n = len(nums)
        low, high = 0, n-1
        while low + 1 < high:
            mid = (high + low) // 2
            if mid % 2 == 0:
                if mid + 1 <= n - 1 and nums[mid] != nums[mid + 1]:
                    high = mid
                else:
                    low = mid
            else:
                if mid - 1 >= 0 and nums[mid] != nums[mid - 1]:
                    high = mid
                else:
                    low = mid

        if low % 2 == 0:
            if nums[low] != nums[low + 1] and low + 1 <= n - 1:
                return nums[low]
            else:
                return nums[high]
        else:
            if nums[low] != nums[low - 1] and low - 1 >= 0:
                return nums[low]
            else:
                return nums[high]



    # 838 Subarray sum equals k
    # find the indices of subarray
    # prefix sum as key, count as value in a dictionary 当prefix_sum - k 出现在HashMap中，叠加count
    def subarraySumEqualsK(self, nums, k):
        count = 0
        prefixSum = 0
        dic = {0: 1}

        for num in nums:
            prefixSum += num
            if prefixSum - k in dic:
                count += dic[prefixSum - k]
            if prefixSum in dic:
                dic[prefixSum] += 1
            else:
                dic[prefixSum] = 1
        return count



    # 397 · Longest Continuous Increasing Subsequence
    # An increasing continuous subsequence:
    # Can be from right to left or from left to right.
    # Indices of the integers in the subsequence should be continuous
    def longestIncreasingContinuousSubsequence(self, A):
        size = len(A)
        if size < 1: return 0
        if size < 2: return 1

        dp1, dp2 = 1, 1
        globalMax = 0

        for i in range(1, size):
            dp1 = dp1 + 1 if A[i] > A[i - 1] else 1
            dp2 = dp2 + 1 if A[i] < A[i - 1] else 1
            globalMax = max(globalMax, max(dp1, dp2))

        return globalMax


    # 1901 · Squares of a Sorted Array
    def SquareArray(self, A: List[int]) -> List[int]:
        return sorted([x**2 for x in A])


    # 149 · Best Time to Buy and Sell Stock
    def maxProfit(self, prices):
        if len(prices) == 0: return 0

        min_price = prices[0]
        max_profit = 0

        for price in prices:
            min_price = min(price, min_price)
            max_profit = max(price - min_price, max_profit)

        return max_profit


    # 402 · Continuous Subarray Sum 
    # find a continuous subarray where the sum of numbers is the biggest. 
    # Your code should return the index of the first number and the index of the last number. 
    # (If their are duplicate answer, return the minimum one in lexicographical order)
    # Greedy
    def continuousSubarraySum(self, A):
        n = len(A)
        max_sum = A[0]  # 全局最大连续子数组和
        sum = 0  # 当前元素必选的最大和
        first = last = 0  # first, last表示全局最大连续子数组的左右端点
        begin = 0  # 当前连续子数组的左端点

        for i in range(n):
            if sum >= 0:
                sum += A[i]
            else:
                begin = i
                sum = A[i]
            if max_sum < sum:
                max_sum = sum
                first = begin
                last = i

        return [first, last]

    # 412 Candy 
    # Each child must have at least one candy.
    # Children with a higher rating get more candies than their neighbors.
    # Input: [1, 2, 2] Output: 4
    def candy(self, ratings):
        n = len(ratings)
        candyNum = [1] * n

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candyNum[i] = candyNum[i-1] + 1
        for i in range(n - 2, -1, -1):     # range(start, stop, step)
            if ratings[i+1] < ratings[i] and candyNum[i+1] >= candyNum[i]:
                candyNum[i] = candyNum[i+1] + 1
        
        return sum(candyNum)


    # 1393 · Friends Of Appropriate Ages
    def numFriendRequests(self, ages):
        def request(a, b):
            return not (b <= 0.5 * a + 7 or b > a or b > 100 and a < 100)

        c = collections.Counter(ages)
        result = 0

        for a in c.keys():
            for b in c.keys():
                if request(a, b):
                    if a != b:
                        result += c[a] * c[b]
                    else:
                        result += c[a] * (c[b] - 1)
                
        return result


    # 148 Sort Colors 
    # Given an array with n objects colored red, white or blue, 
    # sort them so that objects of the same color are adjacent, 
    # with the colors in the order red 0, white 1 and blue 2. 
    # left 的左侧都是 0（不含 left）
    # right 的右侧都是 2（不含 right）
    # index 从左到右扫描每个数，如果碰到 0 就丢给 left，碰到 2 就丢给 right。碰到 1 就跳过不管
    def sortColors(self, nums):
        left, index, right = 0, 0, len(nums) - 1
        while index <= right:
            if nums[index] == 0:
                nums[left] , nums[index] = nums[index], nums[left]
                left += 1
                index += 1
            elif nums[index] == 2:
                nums[right] , nums[index] = nums[index], nums[right]
                right -= 1
            else:
                index += 1


    # 539 · Move Zeroes 
    # move all 0's to the end of it while maintaining the relative order of the non-zero elements
    def moveZeroes(self, nums):
        index = 0
        for num in nums:
            if num != 0:
                nums[index] = num
                index += 1
            for i in range(index, len(nums)):
                nums[i] = 0
        return nums


    # 767 · Reverse Array
    def reverseArray(self, nums):
        i, j = 0, len(nums) - 1
        while i < j:
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
            i += 1
            j -= 1
        return nums



    # 41 · Maximum Subarray 
    # Given an array of integers, find a contiguous subarray which has the largest sum.
    # --------- prefix sum method ----------
    # 根据公式 Sum[i..j] = PrefixSum[j] - PrefixSum[i - 1]，
    # 在 PrefixSum[j] 固定的情况下，为了使 Sum[i..j] 最大，PrefixSum[i - 1] 需要最小，
    # 所以需要记录已经遍历的数字中最小的 PrefixSum。
    def maxSubArrayPrefix(self, nums):
        min_prefix, curr_prefix = 0, 0
        max_result = - sys.maxsize - 1

        for num in nums:
            curr_prefix += num
            max_result = max(max_result, curr_prefix - min_prefix)
            min_prefix = min(min_prefix, curr_prefix)
        
        return max_result

    # 41 · Maximum Subarray 
    # Given an array of integers, find a contiguous subarray which has the largest sum.
    # --------- greedy method ----------
    def maxSubArrayGreedy(self, nums):
        # max_result记录全局最大值 curr_sum记录当前子数组的和
        max_result = - sys.maxsize - 1
        curr_sum = 0

        for num in nums:
            curr_sum += num
            max_result = max(max_result, curr_sum)
            curr_sum = max(curr_sum, 0)
        
        return max_result

    # 41 · Maximum Subarray 
    # Given an array of integers, find a contiguous subarray which has the largest sum.
    # --------- DP method ----------
    def maxSubArrayDP(self, nums):
        dp = [0 for x in range(len(nums))]
        dp[0] = nums[0]
        for i in range(1, len(nums)):
            dp[i] = max(0, dp[i-1] + nums[i])
        max_result = -sys.maxsize - 1
        for i in dp:
            max_result = max(max_result, i)
        return max_result

####################### Two Pointers ###########################

    # 891 · Valid Palindrome II
    # Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.
    def validPalindrome(self, s):
        left = 0
        right = len(s) - 1
        while left + 1 < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.validPalindromeHelper(s, left, right - 1) or self.validPalindromeHelper(s, left + 1, right)

        return True
    
    def validPalindromeHelper(self, s, left, right):
        while left + 1 < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True


    # 3sum
    # Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0
    # target = -a， 找到满足 b + c = target 的数字
    def threeSum(self, numbers):
        if not numbers or len(numbers) < 3:
            return []
        
        result = []

        for i in range(len(numbers) - 2):
            target = - numbers[i]
            dict = {}
            for j in range(i+1, len(numbers)):
                if target - numbers[j] in dict:
                    temp_result = sorted([numbers[i], numbers[j], target - numbers[j]])
                    if temp_result not in result:
                        result.append(temp_result)
                else:
                    dict[numbers[j]] = j

        return result



####################### DP ###########################

    # 669 · Coin Change
    # use the coins values in given array to find min number of coins needed to add up to target
    # （1）确定状态：dp[i]表达要凑出i元钱可拥有的方案数。所以如果最后一枚硬币面值为c，那么求出dp[i - c]的方案数即可推出dp[i]。
    # （2）转移方程：假设硬币面值为序列c[n]，dp[i] = min(dp[i - c0] + 1, dp[i - c1] + 1, ... , dp[i - cn] + 1)
    # （3）初始状态和边界情况：dp[0] = 0，对于凑不出的i值，dp[i]值应该为sys.maxsize
    # （4）计算顺序：从左往右。
    # 时间复杂度：O(n * m)，n为目标钱数，m为硬币面值种类数。空间复杂度：O(n)。
    # import sys
    def coinChange(self, coins, amount):
        if not coins or len(coins) == 0 or amount < 0:
            return -1

        n = len(coins)
        dp = [sys.maxsize] * (amount + 1) # use amount + 1 to prevent overflow
        dp[0] = 0

        for i in range(1, amount + 1):
            for j in range(n):
                last_index = i - coins[j]
                if last_index < 0 or dp[last_index] == sys.maxsize:
                    continue
                dp[i] = min(dp[i], dp[last_index] + 1)

        if dp[amount] == sys.maxsize:
            return -1
        
        return dp[amount]




    # 1702 · Distinct Subsequences II 
    # Given a string S, count the number of distinct, non-empty subsequences of S .
    # def distinctSubseqII(self, S):
    #    n = len(S)
        




####################### Graph ###########################

    # 1031 · Is Graph Bipartite?
    # a undirected graph is bipartite if we can split it's set of nodes into two independent subsets A and B 
    # such that every edge in the graph has one node in A and another node in B.
    # 二分图染色模板题,通过黑白染色我们可以判断一个无向图是否二分图:
    # 遍历整个图, 将相邻的节点染成不同的颜色, 如果可以完成这个遍历(即染色过程没有冲突), 说明是二分图.
    # 可以用BFS或DFS来实现, 只需要根据当前节点的颜色设定下一个节点的颜色即可, 如果下一个节点已经被染成了相同的颜色, 说明发生了冲突.
    # Use DFS, time O(v+e) space O(v)
    # 0 means not visited/colored, c means blue, -c means red
    def isBipartite(self, graph):
        n = len(graph)
        self.color = [0] * n
        for i in range(n):
            if self.color[i] == 0 and not self.isBipartiteHelper(i, graph, 1):
                return False
        return True

    def isBipartiteHelper(self, now, graph, c):
        self.color[now] = c
        for nxt in graph[now]:
            if self.color[nxt] == 0 and not self.isBipartiteHelper(nxt, graph, -c):
                return False
            elif self.color[nxt] == self.color[now]:
                return False
        return True



    # 137 · Clone Graph
    class UndirectedGraphNode:
        def __init__(self, x):
            self.label = x
            self.neighbors = []

    def cloneGraph(self, node):
        if not node:
            return None

        # 1. find nodes
        nodes = self.findNodeBFS(node)
        # 2. copy nodes
        mapping = self.copyNode(nodes)
        # 3. copy edges
        self.copyEdges(nodes, mapping)

        return mapping[node]

    def findNodeBFS(self, node):
        queue = collections.deque([node])
        visited = set([node])
        while queue:
            curr = queue.popleft()
            for neighbor in curr.neighbors:
                if neighbor in visited:
                    continue
                queue.append(neighbor)
                visited.add(neighbor)

        return list(visited)

    def copyNode(self, nodes):
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        return mapping

    def copyEdges(self, nodes, mapping):
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)


####################### Meeting Room ###########################

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
# 920 · Meeting Rooms
# determine if a person could attend all meetings
def canAttendMeetings(self, intervals):
    # sort by start time
    intervals = sorted(intervals, key = lambda x : x.start)

    for i in range(len(intervals) - 1):
        if intervals[i].end > intervals[i+1].start:
            return False
    
    return True