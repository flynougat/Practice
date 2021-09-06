import java.util.*;

public class LintCode {
    /* **********************************************************************************
       HashMap HashSet TreeSet
       **********************************************************************************
     */

    /*
    HashSet HastMap
    # 1369 most common word
     */
    public String mostCommonWord(String paragraph, String[] banned) {
        // store banned word in set
        Set<String> bannedSet = new HashSet<>();
        for (String s : banned) {
            bannedSet.add(s.toLowerCase());
        }
        //get each word from the paragraph.
        // for ASCII only,to include any other separators between words (like commas and semicolons
        String [] words = paragraph.split("\\W+");

        //initialize variables
        int count = 0;
        String result = "";

        //store word and count in map
        Map<String, Integer> wordCount = new HashMap<>();
        for (String w : words) {
            String wLower = w.toLowerCase();
            if (!bannedSet.contains(wLower)) {
                wordCount.put(wLower, wordCount.getOrDefault(wLower, 0) + 1);
                if (count < wordCount.get(wLower)) {
                    count = wordCount.get(wLower);
                    result = wLower;
                }
            }
        }
        return result;
    }


    /*
    two sum hashMap method
     */
    public int[] twoSumHashMap(int[] numbers, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int number : numbers){
            if(!map.containsKey(number)){
                map.put(number, 1);
            }else {
                map.put(number, map.get(number) +1);
            }
        }
        int a = 0, b = 0;
        for (int key : map.keySet()) {
            int missing = target - key;
            if (map.containsKey(missing)) {
                if (missing != key) {
                    a = key;
                    b = missing;
                }
                if (missing == key && map.get(missing) > 1){
                    a = key;
                    b = missing;
                }

            }
        }
        int count = 0;
        int[] result = new int[2];

        if (a != b){
            HashMap<Integer, Integer> mapIndex = new HashMap<>();
            for(int number : numbers){
                mapIndex.put(number, count++);
            }
            result[0] = mapIndex.get(a);
            result[1] = mapIndex.get(b);
        }

        int i, j;
        if(a == b){
            for(i = 0; i < numbers.length; i++){
                if(a == numbers[i]){
                    result[0] = i;
                    break;
                }
            }
            for(j = i+1; j < numbers.length; j++){
                if(b == numbers[j]){
                    result[1] = j;
                }

            }
        }
        Arrays.sort(result);
        return result;
    }


    /*
    # 1592 Find and Replace Pattern
     */
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> result = new ArrayList<>();
        String newP = transform(pattern);

        int patternLen = pattern.length();
        for (String w : words) {
            int wLen = w.length();
            if (wLen != patternLen) {
                continue;
            }

            String current = transform(w);
            if (current.equals(newP)) {
                result.add(w);
            }
        }
        return result;
    }

    //transform abc abb patterns to numbers like 123 122
    private String transform(String word) {
        int label = 0;
        HashMap<Character, Integer> map = new HashMap<>();
        char[] cs = word.toCharArray();
        for (int i = 0; i < cs.length; i++) {
            char c = cs[i];
            if (map.containsKey(c)) {
                cs[i] = (char) (map.get(c) + '0');
            }else {
                map.put(c, label);
                cs[i] = (char) (label + '0');
                label++;
            }
        }
        String p = new String(cs);
        return p;
    }


    /*
    HARD - Revisit
    https://www.youtube.com/watch?v=gD4dzeQ6YH0
    #1278 Max Sum of Rectangle No Larger Than K
    */
    public int maxSumSubmatrix(int[][] matrix, int k) {
        int m = matrix.length, n = matrix[0].length;

        int max = Integer.MIN_VALUE;

        for (int i=0; i<m; i++){
            int[] add = new int[n];
            for (int j = i; j<m; j++){
                sum(add, matrix[j]);

                TreeSet<Integer> treeset = new TreeSet<>();
                max = Math.max(max, helper(add, treeset, k));
                if (max == k){
                    return max;
                }
            }
        }
        return max == Integer.MIN_VALUE ? -1 : max;
    }
    private int helper(int[] add, TreeSet<Integer> treeset, int k){
        treeset.add(0);
        int prefixSum = 0;
        int curMax = Integer.MIN_VALUE;
        for (int ele : add) {
            prefixSum += ele;
            Integer ceil = treeset.ceiling(prefixSum - k);
            if (ceil != null){
                if (prefixSum - ceil == k){
                    return k;
                }else {
                    curMax = Math.max(curMax, prefixSum - ceil);
                }
            }
            treeset.add(prefixSum);
        }
        return curMax;
    }
    private void sum(int[] add, int[] cols){
        for (int i=0; i<cols.length; i++){
            add[i] += cols[i];
        }
    }


    /*
     # 488 Happy Number
     HashSet to check if the number already exist. If so, break the loop and return false
     */
    public boolean isHappy(int n) {
        HashSet<Integer> seen = new HashSet<>();
        while (n != 1){
            int current = n;
            int sum = 0;
            while (current != 0){
                sum += (current % 10) * (current % 10);
                current /= 10;
            }

            if (seen.contains(sum)) return false;

            seen.add(sum);
            n = sum;
        }
        return true;
    }





    /* ***********************************************************************************
       Two Pointer
       ***********************************************************************************
     */

    /*
     # 13 · Implement strStr()
     two pointer
     */
    public int strStr(String source, String target) {
        for (int i = 0; i < source.length() - target.length() + 1; i++){
            int j;
            for (j = 0; j < target.length(); j++) {
                if (source.charAt(i+j) != target.charAt(j)) {
                    break;
                }
            }
            if (j == target.length()){
                return i;
            }
        }
        return -1;
    }

    /*
    #56 two sum
    check if two numbers in an arr added up to target
    assume only one solution, two pointer
     */

    public int[] twoSum(int[] numbers, int target) {
        // must copy original arr to new one.
        // find if the 2 numbers exist
        // find the index of 2 number in original arr
        int[] numbersCopy = Arrays.copyOf(numbers, numbers.length);
        Arrays.sort(numbersCopy);

        int index1 = 0;
        int index2 = numbersCopy.length - 1;
        int a = 0;
        int b = 0;
        int[] result = new int[2];
        while (index1 < index2) {
            if (numbersCopy[index1] + numbersCopy[index2] > target){
                index2--;
            }else if (numbersCopy[index1] + numbersCopy[index2] < target){
                index1++;
            }else {
                a = numbersCopy[index1];
                b = numbersCopy[index2];
                break;
            }
        }

        int i;
        if(a!=b){
            for(i = 0; i < numbers.length; i++){
                if(a == numbers[i]){
                    result[0] = i;
                }
                if(b == numbers[i]){
                    result[1] = i;
                }
            }
        }

        if(a == b){
            for(i = 0; i < numbers.length; i++){
                if(a == numbers[i]){
                    result[0] = i;
                    break;
                }
            }
            for(int j = i+1; j < numbers.length; j++){
                if(b == numbers[j]){
                    result[1] = j;
                }

            }
        }
        Arrays.sort(result);
        return result;
    }


    /*
    # 928 Longest Substring with at most 2 distinct characters
    Input: “eceba”
    Output: 3
    sliding window problem, use two pointers & HashMap
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int len = s.length();
        if (len < 3) return len;

        int aPointer = 0;
        int bPointer = 0;
        int max = 2;

        HashMap<Character, Integer> map = new HashMap<>();

        while (bPointer < len) {
            if (map.size() < 3) {
                map.put(s.charAt(bPointer), bPointer);
                bPointer++;
            }
            if(map.size() > 2) {
                //remove the left most
                int delIndex = Collections.min(map.values());
                map.remove(s.charAt(delIndex));
                aPointer = delIndex + 1;
            }
            max = Math.max(max, bPointer - aPointer);
        }
        return max;
    }

    /*
    #363 trap rain water
    Two Pointer
     */
    public int trapRainWater(int[] heights) {
        // min(L, R) - height[i], sum all positive
        //two pointer time O(n), space O(1)
        if (heights.length == 0) return 0;
        int left = 0;
        int right = heights.length - 1;
        int leftMax = heights[left];
        int rightMax = heights[right];
        int result = 0;

        while (left < right){
            if (leftMax < rightMax) {
                left++;
                leftMax = Math.max(leftMax, heights[left]);
                result += leftMax - heights[left];
            }else {
                right--;
                rightMax = Math.max(rightMax, heights[right]);
                result += rightMax - heights[right];
            }
        }
        return result;
    }




    /* ***********************************************************************************
       Sort
       ***********************************************************************************
     */

    /*
    #64 easy Merge Sort
    Given two sorted integer arrays A and B, merge B into A as one sorted array.
     */
    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        for (int i = m, j = 0; j < n; i++, j++) {
            A[i] = B[j];
        }
        Arrays.sort(A);
    }




    /* ***********************************************************************************
       Binary Tree
       ***********************************************************************************
     */

    /*
    # 1172 Binary tree tilt
    use recursion
     */
    public class TreeNode {
        public int val;
        public TreeNode left, right;

        public TreeNode(int val) {
            this.val = val;
            this.left = this.right = null;
        }
    }

    int result = 0;

    public int findTilt(TreeNode root) {
        tilt(root);
        return result;
    }
    private int tilt(TreeNode root) {
        if(root == null) return 0;
        int left = tilt(root.left);
        int right = tilt(root.right);
        result += Math.abs(left - right);
        return left + right + root.val;
    }



    /*
    #1106 · Maximum Binary Tree
    The root is the maximum number in the array.
    The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
    The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
     */
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        // null case
        if (nums.length == 0) return null;

        //find max
        int index = 0;
        int max = nums[0];
        for (int i = 1; i<nums.length; i++) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }

        //build tree
        TreeNode root = new TreeNode(max);
        root.left = constructMaximumBinaryTree(Arrays.copyOfRange(nums, 0, index));
        if (index == nums.length - 1) {
            root.right = null;
        }
        root.right = constructMaximumBinaryTree(Arrays.copyOfRange(nums, index + 1, nums.length));

        return root;
    }


    /* ********************************************************************************
       Binary Search Tree
       ********************************************************************************
     */

    /*
    # 900 closest binary search tree value
    take advantage of BST, use recursion
     */
    public int closestValue(TreeNode root, double target) {
        return closestValueHelper(root, target, Double.MAX_VALUE, root.val);
    }
    private int closestValueHelper(TreeNode root, double target, double diff, int currClosest) {
        if (root == null) {
            return currClosest;
        }
        if (Math.abs(target - root.val) < diff) {
            diff = Math.abs(target - root.val);
            currClosest = root.val;
        }
        if (target > root.val) {
            return closestValueHelper(root.right, target, diff, currClosest);
        }else {
            return closestValueHelper(root.left, target, diff, currClosest);
        }
    }



    /* *********************************************************************************
       BST
       *********************************************************************************
     */


    /*
       # 69 · Binary Tree Level Order Traversal
       BST
    */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List result = new ArrayList();

        if (root == null) return result;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()){
            List level = new ArrayList();
            int size = queue.size();
            for (int i = 0; i < size; i++){
                TreeNode head = queue.poll();
                level.add(head.val);
                if (head.left != null){
                    queue.offer(head.left);
                }
                if (head.right != null){
                    queue.offer(head.right);
                }
            }
            result.add(level);
        }
        return result;
    }

    private int treeHeight (TreeNode root){
        if (root == null) return 0;

        int lHeight = treeHeight(root.left);
        int rHeight = treeHeight(root.right);

        //return the larger number + root level
        if (lHeight > rHeight) {
            return lHeight + 1;
        }else {
            return rHeight +1;
        }
    }





    /* *********************************************************************************
       Heap
       *********************************************************************************
     */

    /*
      # 919 Meeting room II
      1. sort intervals by starting time
      2. keep track of ending
    */
    public int minMeetingRooms(List<Interval> intervals) {
        //no meeting case
        if (intervals == null || intervals.size() == 0) {
            return 0;
        }
        Collections.sort(intervals, (a, b) -> a.start - b.start);
        PriorityQueue<Interval> minHeap = new PriorityQueue<>((a,b) -> a.end - b.end);
        minHeap.add(intervals.get(0));
        for (int i = 1; i < intervals.size(); i++) {
            Interval curr = intervals.get(i);
            Interval earliest = minHeap.remove();
            if (curr.start >= earliest.end) {
                earliest.end = curr.end;
            }else {
                minHeap.add(curr);
            }
            minHeap.add(earliest);
        }
        return minHeap.size();
    }

    public class Interval {
        int start, end;
        Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }
    }




    /* ****************************************************************************
       List
       ****************************************************************************
    */


    /*
       # 174 remove nth Node from end of list
       method 1: pointers
       method 2: copy linked list to arraylist, then remove the item from arraylist,
       re-establish linked list
    */
    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
            next = null;
        }
    }
    public ListNode removeNthFromEnd(ListNode head, int n) {
        /*
        method 1
        - faster pointer goes to n+1 position
        - slow pointer goes from the beginning to the position before target
         */
        //create dummy head to quickly find head
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode slow = dummyHead;
        ListNode fast = dummyHead;

        for (int i = 0; i <= n+1; i++) {
            fast = fast.next;
        }
        while (fast != null){
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return dummyHead.next;

        //method 2
        /*
        if (n==0) return head;

        List<Integer> holder = new ArrayList<>();
        while (head != null) {
            holder.add(head.val);
            head = head.next;
        }
        if (holder.size() == 1) {
            return null;
        }
        holder.remove(holder.size() - n);
        ListNode newHead = new ListNode(holder.get(0));
        ListNode result = newHead;
        for (int i = 1; i < holder.size(); i++) {
            newHead.next = new ListNode(holder.get(i));
            newHead = newHead.next;
        }
        return newHead;
        */
    }


    /*
    #599 insert into a cyclic sorted list
     */
    public ListNode insert(ListNode node, int x) {
        // head is null
        if (node == null) {
            ListNode head = new ListNode(x);
            head.next = head;
            return head;
        }

        ListNode max = node;
        while (max.next != node && max.val <= max.next.val) {
            max = max.next;
        }
        ListNode min = max.next;
        ListNode curr = min;
        if (x >= max.val || x <= min.val){
            ListNode head = new ListNode(x);
            max.next = head;
            head.next = min;
        }else {
            while (curr.next.val < x){
                curr = curr.next;
            }
            ListNode head = new ListNode(x);
            head.next = curr.next;
            curr.next = head;
        }
        return node;
    }


    /*
    # 104 · Merge K Sorted Lists
    use minHeap time O(nlogk) space O(N)
     */

    public ListNode mergeKLists(List<ListNode> lists) {
        //construct a minHeap
        PriorityQueue<Integer> minheap = new PriorityQueue<>();
        for (ListNode head : lists) {
            while (head != null) {
                minheap.add(head.val);
                head = head.next;
            }
        }
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;

        //move items from minHeap
        while (!minheap.isEmpty()){
            head.next = new ListNode(minheap.remove());
            head = head.next;
        }

        return dummy.next;
    }


    /*
    # 165 · Merge Two Sorted Lists
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) return null;

        ListNode dummy = new ListNode(-1);
        ListNode tail = dummy;

        while (l1 != null && l2 != null){
            if (l1.val < l2.val){
                tail.next = l1;
                l1 = l1.next;
            }else {
                tail.next = l2;
                l2 = l2.next;
            }
            tail = tail.next;
        }

        //handle remaining
        if (l1 != null){
            tail.next = l1;
        }
        if (l2 != null) {
            tail.next = l2;
        }
        return dummy.next;
    }


    /*
    # 756 · Multiply Two Numbers
    Given two numbers represented by linked lists, write a function that
    returns the multiplication of these two linked lists.
    Example:
    Input：3->2->1->null,1->2->null
    Output：3852
    Explanation：321*12=3852
    traverse each linked list. when move to next node, multiply previous value with 10
     */
    public long multiplyLists(ListNode l1, ListNode l2) {
        ListNode temp = new ListNode(0);
        temp = l1;
        long num1 = 0;

        while (temp != null){
            num1 *= 10;
            num1 += temp.val;
            temp = temp.next;
        }

        temp = l2;
        long num2 = 0;

        while (temp != null){
            num2 *= 10;
            num2 += temp.val;
            temp = temp.next;
        }

        return num1 * num2;
    }


    /*
    #374 Spiral matrix
    Array, List
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();

        //special case
        if (matrix.length == 0) return result;

        int rawBegin = 0;
        int rawEnd = matrix.length - 1;
        int columnBegin = 0;
        int columnEnd = matrix[0].length - 1;

        while(rawBegin <= rawEnd && columnBegin <= columnEnd) {
            // traverse raw by column index left to right
            for (int i = columnBegin; i <= columnEnd; i++){
                result.add(matrix[rawBegin][i]);
            }
            rawBegin++;

            //traverse column by raw position up to bottom
            for (int i = rawBegin; i <= rawEnd; i++) {
                result.add(matrix[i][columnEnd]);
            }
            columnEnd--;

            //check bottom raw, traverse from right to left
            if (rawBegin <= rawEnd){
                for (int i=columnEnd; i >= columnBegin; i--){
                    result.add(matrix[rawEnd][i]);
                }
            }
            rawEnd--;

            //traverse column bottom to up
            if (columnBegin <= columnEnd){
                for (int i = rawEnd; i >= rawBegin; i--){
                    result.add(matrix[i][columnBegin]);
                }
            }
            columnBegin++;
        }
        return result;
    }


    /*
    # 30 Insert Interval
    Original sorted bt staring point, non-overlapping
    Insert a new interval into it, make sure the list is still in order and non-overlapping
    (merge intervals if necessary).
     */
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new ArrayList<>();

        //special case
        if (intervals.size() == 0 || intervals == null) {
            result.add(newInterval);
        }

        for (int i = 0; i < intervals.size(); i++){
            //insert before the ith interval, add the remaining
            if (newInterval.end < intervals.get(i).start){
                result.add(newInterval);
                for (int j = i; j<intervals.size(); j++){
                    result.add(intervals.get(j));
                }
                break;
            }
            else if (newInterval.start > intervals.get(i).end){
                result.add(intervals.get(i));
            }else {
                newInterval.start = Math.min(newInterval.start, intervals.get(i).start);
                newInterval.end = Math.max(newInterval.end, intervals.get(i).end);
            }
            //add to last
            if (i == intervals.size() - 1){
                result.add(newInterval);
            }
        }
        return result;
    }


    /*
    #425 Letter combinations of phone
    Linked List
     */
    public List<String> letterCombinations(String digits) {
        //null exception
        LinkedList<String> result = new LinkedList<>();
        if (digits.length() == 0) return result;

        result.add("");

        String[]char_map = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs",
                "tuv", "wxyz"};

        //store in queue
        for (int i = 0; i < digits.length(); i++){
            int index = Character.getNumericValue(digits.charAt(i));
            while (result.peek().length()==i) {
                String permutation = result.remove();
                for (char c : char_map[index].toCharArray()) {
                    result.add(permutation + c);
                }
            }
        }
        return result;
    }


    /*
    # 57 · 3Sum = 0
    sort, point at an element, look for 2sum from the rest
    No duplicates allowed. if nums[index] == nums[index -1] --> skip
    time O(n^2)
     */
    public List<List<Integer>> threeSum(int[] numbers) {
        // write your code here
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(numbers);
        for (int i = 0; i < numbers.length - 2; i++){
            if (numbers[i] > 0) break; //sorted in ascending order, can't find negative to add
            if (i > 0 && numbers[i] == numbers[i-1]) continue; //skip to avoid duplicates

            int target = 0 - numbers[i];

            int low = i+1, high = numbers.length - 1;
            while (low < high) {
                int tempSum = numbers[low] + numbers[high];
                if (tempSum == target) {
                    result.add(Arrays.asList(new Integer[]{numbers[i], numbers[low], numbers[high]}));
                    low++;
                    while (low < high && numbers[low] == numbers[low-1]) low++;

                    high--;
                    while (low < high && numbers[high] == numbers[high+1]) high--;
                }else if (tempSum < target) {
                    low++;
                }else {
                    high--;
                }
            }
        }
        return result;
    }



    /* ************************************************************************
       Array
       ************************************************************************
     */

    /*
    #100 remove duplicates from sorted array and return the remaining length
     */
    public int removeDuplicates(int[] nums) {
        //exception
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return 1;
        }
        int index = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] != nums[i+1]) {
                nums[index++] = nums[i+1];
            }
        }
        return index;
    }


    /*
    #433 Number of Islands
    use matrix row and column as unique Key for 1, in i-j pair format
     */
    public int numIslands(boolean[][] grid) {
        if (grid == null || grid.length ==0) return 0;

        int result = 0;

        for (int i = 0; i<grid.length; i++){
            for (int j = 0; j < grid[0].length; j++){
                if (grid[i][j] == true) {
                    result += islandDFS(grid, i, j);
                }
            }
        }
        return result;
    }

    private int islandDFS(boolean[][] grid, int i, int j){
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == false){
            return 0;
        }
        grid[i][j] = false;
        islandDFS(grid, i+1, j);
        islandDFS(grid, i-1, j);
        islandDFS(grid, i, j+1);
        islandDFS(grid, i, j-1);
        return 1;
    }


    /*
    # 52 Next Permutation
    Step 1 find change position
    Start from end to front, scan items and stop at first position where nums[p] < nums[p + 1]
    step 2 find the number for substitution
    Scan from right to left and stop at the first element that is greater than num[p], record the index q.
    step 3 swap num[p] and num[q]
    step 4 re-arrange items after position p in increasing order to make the remaining minimum
     */
    public int[] nextPermutation(int[] nums) {
        // special case
        if (nums.length <=1) return nums;

        //step1
        int p = nums.length - 2;
        while (p >= 0 && nums[p] >= nums[p+1]) {
            p--;
        }
        //step 2 & 3
        int q = nums.length - 1;
        if (p >= 0) {
            while (q >= 0 && nums[q] <= nums[p]) q--;
            swap (nums, p, q);
        }
        //step 4
        reverse(nums, p+1);

        return nums;
    }

    private void swap(int[] nums, int p, int q) {
        int temp = nums[p];
        nums[p] = nums[q];
        nums[q] = temp;
    }

    private void reverse (int[] nums, int start){
        int end = nums.length - 1;
        while (start < end) {
            swap(nums, start, end);
            start ++;
            end --;
        }
    }



    /*
    # 62 · Search in Rotated Sorted Array
    a sorted array is rotated at some pivot unknown to you beforehand.
    You are given a target value to search. If found in the array return its index, otherwise return -1.
    Use modified binary search to find pivot
    4 5 6 7 0 1 2
     */
    public int search(int[] A, int target) {
        //exception
        if (A == null || A.length == 0) return -1;

        //find smallest element use modified binary search
        int left = 0;
        int right = A.length - 1;

        while (left + 1 < right) {
            int mid = left + (right - left)/2;

            //left side is sorted, use regular binary search
            if(A[left] < A[mid]){
                if (A[left] <= target && target < A[mid]){
                    right = mid;
                }else {
                    left = mid;
                }
            }else {
                if (A[mid] < target && target <= A[right]){
                    left = mid;
                }else {
                    right = mid;
                }
            }
        }
        if (A[left] == target) return left;
        if (A[right] == target) return right;
        return -1;
    }




    /* ************************************************************************
       DP + BFS
       ************************************************************************
     */

    /*
    HARD - revisit
    #1422 Shortest Path Visiting All Nodes
    use DP + BFS. time O(n*2^n), space O(n*2^n)
     */
    public int shortestPathLength(int[][] graph) {
        final int V = graph.length;
        int[][] dp = new int[V][1<<V];
        for (int[] row : dp){
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        Queue<int[]> q = new ArrayDeque<>();
        //start
        for (int v = 0; v!=V; v++){
            dp[v][1<<v] = 0;
            q.offer(new int[] {v, 1<<v});
        }
        for (int step = 1; !q.isEmpty(); step++){
            for (int b = q.size(); b != 0; b--){
                int[] now = q.poll();
                final int at = now[0], start = now[1];

                //to next position
                for (int next : graph[at]){
                    final int nextStart = start | (1<<next);
                    if (dp[next][nextStart] != Integer.MAX_VALUE)
                        continue;
                    dp[next][nextStart] = step;
                    q.offer(new int[]{next, nextStart});
                }
            }
        }

        int res = Integer.MAX_VALUE;
        for (int v=0; v!=V; v++) {
            res = Math.min(res, dp[v][(1<<V) - 1]);
        }

        return res;
    }


    /*
       117 · Jump Game II
       greedy BFS time O(N)
       DP time O(N^2)
       think of most far can reach
     */
    public int jump(int[] A) {
        if (A == null || A.length == 0) {
            return -1;
        }

        int start = 0, end = 0, farthest = 0;
        int result = 0;

        for (int i = 0; i < A.length - 1; i++){
            farthest = Math.max(farthest, A[i] + i);
            if (i == end) {
                result++;
                end = farthest;
            }
        }
        return result;
    }






    /* ************************************************************************
       String
       ************************************************************************
     */


    /*
    # 188 Insert 5, return the largest after insertion
    - if positive number, insert before the first number < 5
    - if negative number, insert before the first number > 5
    */
    public int InsertFive(int a) {
        String aString = String.valueOf(a);
        int i = 0;
        if (a >= 0) {
            while (i<aString.length() && aString.charAt(i) < '5'){
                i++;
            }
        }else {
            i = 1;
            while (i<aString.length() && aString.charAt(i) > '5'){
                i++;
            }
        }
        int result = Integer.parseInt(aString.substring(0, i) + '5' + aString.substring(i));
        return result;
    }


    /*
    # 408 add binary
    Given two binary strings, return their sum (In binary notation).
     */
    public String addBinary(String a, String b) {
        // write your code here
        int numberA = Integer.parseInt(a, 2);
        int numberB = Integer.parseInt(b, 2);

        int sum = numberA + numberB;

        return Integer.toBinaryString(sum);
    }


    /*
    #655 Add Strings
     */
    public String addStrings(String num1, String num2) {
//        int a = Integer.parseInt(num1);
//        int b = Integer.parseInt(num2);
//        int sum = a + b;
//        return String.valueOf(sum);

        StringBuilder result = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() -1;
        int carry = 0;
        while (i>=0 || j>=0){
            int sum = carry;
            if (i>=0){
                sum += num1.charAt(i--) - '0';
            }
            if (j>=0){
                sum += num2.charAt(j--) - '0';
            }
            result.append(sum % 10);
            carry = sum / 10;
        }
        if (carry != 0) result.append(carry);

        return result.reverse().toString();
    }


    /*
    # 1350 · Excel Sheet Column Title
    Given a positive integer, return its corresponding column title as appear in an Excel sheet.
     */
    public String convertToTitle(int n) {
        StringBuilder str = new StringBuilder();
        while (n > 0){
            n--;
            str.append((char) ((n % 26) + 'A'));
            n /= 26;
        }
        return str.reverse().toString();
    }


    /*
    # 107 word break
    Given a string s and a dictionary of words dict,
    determine if s can be broken into a space-separated sequence of one or more dictionary words.
    Use DP
    optimization 1: calculate max word length. when i - j > maxLen, skip
    optimization 2: start j from the right side
     */
    public boolean wordBreak(String s, Set<String> wordSet) {
        boolean[] dp = new boolean[s.length()+1];

        // find maxWordLen
        int maxWordLen = 0;
        for (String word : wordSet){
            maxWordLen = Math.max(maxWordLen, word.length());
        }

        dp[0] = true;
        for (int i = 0; i <= s.length(); i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (i - j > maxWordLen) {
                    continue;
                }
                if (dp[j] && wordSet.contains(s.substring(j, i))){
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /*
     # 422 Length of Last Word
     - Method 1: Use String.split()
     - Method 2: Use trim
     - Method 3: Array, count from tail until " "
     Example
     Input: "Hello World"
     Output: 5
     */
    public int lengthOfLastWord(String s) {
        //Method 1: Use String.split()
//        String[] words = s.split(" ");
//        return words.length == 0 ? 0 : words[words.length - 1].length();

        //Method 2: Use trim
//        return s.trim().length() - s.trim().lastIndexOf(" ") - 1;

        //Method 3: Array
        int length = 0;
        char[] chars = s.toCharArray();
        for (int i = s.length() - 1; i >= 0; i--) {
            if (length == 0) {
                if (chars[i] == ' ') {
                    continue;
                } else {
                    length++;
                }
            } else {
                if (chars[i] == ' ') {
                    break;
                } else {
                    length++;
                }
            }
        }
        return length;
    }




    /* ************************************************************************
       Matrix
       ************************************************************************
     */


    /*
    #28 Search a 2D Matrix
    Integers in each row are sorted from left to right.
    The first integer of each row is greater than the last integer of the previous row.
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        // exception
        if (matrix == null || matrix.length == 0) return false;

        int rowIndex = matrix.length - 1;
        int columnIndex = matrix[0].length - 1;

        // check target at which row
        int start = 0;
        int end = rowIndex;
        while (start + 1 < end) {
            int mid = start + (end - start)/2;
            if (matrix[mid][0] == target) return true;
            if (target < matrix[mid][0]){
                end = mid;
            }else {
                start = mid;
            }
        }
        //handle first or last row
        if (matrix[end][0] <= target){
            rowIndex = end;
        }else if (matrix[start][0] <= target) {
            rowIndex = start;
        }else {
            return false;
        }

        //find column index
        start = 0;
        end = columnIndex;
        while (start + 1 < end) {
            int mid = start + (end - start)/2;
            if (matrix[rowIndex][mid] == target) return true;
            if (matrix[rowIndex][mid] < target) {
                start = mid;
            }else {
                end = mid;
            }
        }
        //check if exist
        if (matrix[rowIndex][start] == target) return true;
        if (matrix[rowIndex][end] == target) return true;
        return false;
    }



    /*
    #737 Find a element that appear in all the rows
     */
    public int FindElements(int[][] Matrix) {
        List<Integer> list = new ArrayList<>();

        int rowLen = Matrix.length;
        int columnLen = Matrix[0].length;

        //loop first row, find possible keys
        for (int i = 0; i < columnLen; i++){
            list.add(Matrix[0][i]);
        }

        for (int i = 1; i < rowLen; i++){
            Set<Integer> set = new HashSet<>();
            for (int j = 0; j < columnLen; j++){
                set.add(Matrix[i][j]);
            }
            for (int j = 0; j < list.size(); j++){
                if (!set.contains(list.get(j))){
                    list.remove(list.get(j));
                }
            }
        }
        return list.get(0);
    }





    /* ****************************************************************************
       Stack
       ****************************************************************************
    */

    /*
    #423 Valid Parentheses
    length should be even number
    pair
     */
    public boolean isValidParentheses(String s) {
        if (s.length() % 2 != 0) return false;
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '['){
                stack.push(c);
            }
            if (c == ')'){
                if(stack.isEmpty()){
                    return false;
                }
                if(stack.peek() != '('){
                    return false;
                }else {
                    stack.pop();
                }
            }
            if (c == '}'){
                if(stack.isEmpty()){
                    return false;
                }
                if(stack.peek() != '{'){
                    return false;
                }else {
                    stack.pop();
                }
            }
            if (c == ']'){
                if(stack.isEmpty()){
                    return false;
                }
                if(stack.peek() != '['){
                    return false;
                }else {
                    stack.pop();
                }
            }
        }
        return stack.isEmpty();
    }



    /*
    193 · Longest Valid Parentheses
    - if current is '(' push to stack
    - if current is ')'
        case 1: stack is empty or ')' push index, this index is not valid
        case 2: stack is '(' this is valid parentheses. Pop '(' so that stack is empty or ')'
     */

    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        int max = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            }else {
                if (stack.isEmpty() || s.charAt(stack.peek()) == ')') {
                    stack.push(i);
                }else {
                    stack.pop();
// the following part is equivalent to (stack.isEmpty() ? -1 : stack.peek())
//                    int compare = 0;
//                    if(stack.isEmpty()){
//                        compare = -1;
//                    }else{
//                        compare = stack.peek();
//                    }
                    max = Math.max(max, i - (stack.isEmpty() ? -1 : stack.peek()));
                }
            }
        }
        return max;
    }



    /* ****************************************************************************
       Backtracking / DFS
       ****************************************************************************
       核心就是 for 循环里面的递归，在递归调用之前「做选择」，在递归调用之后「撤销选择
       用回溯算法解决 求子集（subset），求排列（permutation），求组合（combination）
       时间复杂度都不可能低于 O(N!)，因为穷举整棵决策树是无法避免的。这也是回溯算法的一个特点，不像动态规划存在重叠子问题可以优化，
       回溯算法就是纯暴力穷举，复杂度一般都很高
       1、路径：也就是已经做出的选择。
       2、选择列表：也就是你当前可以做的选择。
       3、结束条件：也就是到达决策树底层，无法再做选择的条件。
       result = []
       def backtrack(路径, 选择列表):
            if 满足结束条件:
            result.add(路径)
            return

       for 选择 in 选择列表:
            做选择
            backtrack(路径, 选择列表)
            撤销选择

       回溯搜索是深度优先搜索（DFS）的一种对于某一个搜索树来说（搜索树是起记录路径和状态判断的作用），
       回溯和DFS，其主要的区别是，回溯法在求解过程中不保留完整的树结构，而深度优先搜索则记下完整的搜索树。
       为了减少存储空间，在深度优先搜索中，用标志的方法记录访问过的状态，这种处理方法使得深度优先搜索法与回溯法没什么区别了。

    */

    /*
    #15 permutations
    Given a list of numbers, return all possible permutations of it. n! possibilities
    Backtracking.
    DFS
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null) return result;
        boolean[] visited = new boolean[nums.length];
        List<Integer> permutation = new ArrayList<>();

        permuteHelper (nums, visited, permutation, result);

        return result;
    }
    private void permuteHelper(int[] nums, boolean[] visited, List<Integer> permutation, List<List<Integer>> result){
        if (nums.length == permutation.size()){
            result.add(new ArrayList<Integer>(permutation));
            return;
        }
        for (int i = 0; i < nums.length; i++){
            if (visited[i]) continue;;

            permutation.add(nums[i]);
            visited[i] = true;
            permuteHelper(nums, visited, permutation, result);
            visited[i] = false;
            permutation.remove(permutation.size() - 1);
        }
    }


    /*
    # 427 Generate Parentheses
    Backtracking
     */
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        generateParenthesisHelper(result, "", 0, 0, n);
        return result;
    }
    private void generateParenthesisHelper(List<String> result, String current, int open, int close, int nPair) {
        if (current.length() == nPair * 2) {
            result.add(current);
            return;
        }
        if (open < nPair){
            generateParenthesisHelper(result, current + "(", open + 1, close, nPair);
        }
        if (close < open){
            generateParenthesisHelper(result, current + ")", open, close + 1, nPair);
        }
    }


    /*
     # 33 N-Queens
     Backtracking
     Try to place Q on each tile and see if it works. If works, recurse through with the updated board and repeat.
     Use column and row indices to check.
     Translate board into a list and add the list to result lists.
     Return result lists
     time and space complexity O(N!)
     */
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> result = new ArrayList<>();

        if (n <=0) return result;

        solveNQueensHelper(result, new ArrayList<>(), n);

        return result;
    }
    private void solveNQueensHelper(List<List<String>> result, List<Integer> cols, int size){
        if (cols.size() == size){
            result.add(draw(cols));
            return;
        }

        //enumerate Q positions, if not valid, continue
        for (int colIndex = 0; colIndex < size; colIndex++){
            if (!isValidQ(cols, colIndex)) continue;

            //if valid, recurse
            cols.add(colIndex);
            solveNQueensHelper(result, cols, size);
            cols.remove(cols.size() - 1);
        }
    }
    //check if tile is valid Q
    private boolean isValidQ(List<Integer> cols, int col){
        int row = cols.size();
        for (int rowIndex = 0; rowIndex < cols.size(); rowIndex++){
            //same col or diagonal, false
            if (cols.get(rowIndex) == col ||
                row + col == rowIndex + cols.get(rowIndex) ||
                    row - col == rowIndex - cols.get(rowIndex)) {
                return false;
            }
        }
        return true;
    }

    //transfer column integer to result string
    private List<String> draw(List<Integer> cols) {
        List<String> resultDraw = new ArrayList<>();
        for (int i = 0; i < cols.size(); i++){
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < cols.size(); j++){
                sb.append(j == cols.get(i) ? 'Q' : '.');
            }
            resultDraw.add(sb.toString());
        }
        return resultDraw;
    }






    /* ****************************************************************************
       Other
       ****************************************************************************
    */

    /*
    # 1347 Factorial Trailing Zeros.
    Calculate n! and track how many zeros at tail
    Count how many 5
     */

    public int trailingZeroes(int n) {
        int result = 0;
        while (n > 0){
            n = n/5;
            result += n;
        }
        return result;
    }















}
