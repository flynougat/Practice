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


####################### Two Pointers ###########################

    # 1712 · Binary Subarrays With Sum
    # prefixSum + dictionary
    def numSubarraysWithSum(self, A, S):
        if not A:
            return 0

        counter = collections.defaultdict(int)
        counter[0] = 1

        prefixSum, result = 0, 0

        for i in A:
            prefixSum += i
            result += counter[prefixSum - S]
            counter[prefixSum] += 1

        return result