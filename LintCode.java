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
            while (i < aString.length() && aString.charAt(i) > '5') {
                i++;
            }
        }else {
            i = 1;
            while (i < aString.length() && aString.charAt(i) <= '5') {
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



}
