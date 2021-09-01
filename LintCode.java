import java.util.*;

public class LintCode {
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
    assume only one solution
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
    #64 easy Merge Sort
    Given two sorted integer arrays A and B, merge B into A as one sorted array.
     */
    public void mergeSortedArray(int[] A, int m, int[] B, int n) {
        for (int i = m, j = 0; j < n; i++, j++) {
            A[i] = B[j];
        }
        Arrays.sort(A);
    }

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
        return result;
    }



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
    # 928 Longest Substring with at most 2 distinct characters
    Input: “eceba”
    Output: 3

    sliding window problem, use two pointers
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

    /*
    #919 Meeting room II
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
    #425 Letter combinations of phone
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
    #374 Spiral matrix
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
    #363 trap rain water
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
    #1523 Partitioning Array
    each subarray length k
    - Each element in the array occurs in exactly one subsquence -> A.length % k == 0
    - All the numbers in a subsequence are distinct -> max duplicate number <= number of sub array
    - Elements in the array having the same value must be in different subsequences
    return ture if it's possible to partition the array satisfies the above conditions
     */
    public boolean PartitioningArray(int[] A, int k) {
        if (A.length == 0) return true;
        if (A.length % k != 0) return false;

        int numSub = A.length / k;
        int count = 0;
        //key is number, value is how many times occurred
        Map<Integer,Integer> map = new HashMap<>();
        for (int i = 0; i < A.length; i++){
            map.putIfAbsent(A[i], 0);
            map.put(A[i], map.get(A[i]) + 1);
            count = Math.max(count, map.get(A[i]));
        }
        return count <= numSub;
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



    /*
    *******************************************************************
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
    ***********************************************
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
    **************************************************************************************
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



    /*
    
     */
















}
