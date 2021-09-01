public class DP {
    /*
    DP is an algorithmic paradigm that solves a given complex problem by
    breaking it into subproblems and storing the results of subproblems to
    avoid computing the same results again.
    1. overlapping subproblems
    2. optimal substructure
    Solution: use memory: memorization(top-down), tabulation(bottom-up)
     */

    /*
    Step 1: visualize examples
    Step 2: find an appropriate subproblem
    Step 3: find relationship among subproblem
    Step 4: generalize the relationship
    Step 5: implement by solving subproblems in order
     */

    /*
    overlapping subproblems property
    fibonacci number using DP top down
     */
    public int fibTopDowm (int[] arr, int n){
        if (n < 2){
            return n;
        }
        if (arr[n] == 0) {
            arr[n] = fibTopDowm(arr, n-1) + fibTopDowm(arr, n-2);
        }
        return arr[n];
    }

    /*
    fibonacci number using DP bottom up
     */
    public int fibBottomUp (int n) {
        int[] arr = new int[n+1];
        arr[0] = 0;
        arr[1] = 1;

        for (int i = 2; i < n+1; i++) {
            arr[i] = arr [i - 1] + arr [i - 2];
        }
        return arr[n];
    }

    public int fibBottomUpBetter(int n){
        int a = 0;
        int b = 1;
        int c;

        for (int i = 2; i < n+1; i++){
            c = a + b;

            a = b;
            b = c;
        }
        return b;
    }


    /*
    Total number of ways to reach Nth stair
    can move 1,2 or 3 steps at a time
    - go up 0 step, 1 possibility
    - go up 1 step, 1 possibility
    - go up 2 step, 2 possibility
    - go up 3 step, 4 possibility
    Pattern: T(3) = T(2)+T(1)+T(1)
    General: T(n) = T(n-1)+T(n-2)+T(n-3)
     */
    // regular recursion method
    public int totalWaysToStair(int n) {
        if (n == 0) return 1; // stay at current position
        if (n == 1) return 1;
        if (n == 2) return 2;

        return totalWaysToStair(n-1)+totalWaysToStair(n-2)+totalWaysToStair(n-3);
    }
    //DP top down method
    public int totalWaysToStairTopDown(int[] arr, int n) {
        if (n == 0) return 1; // stay at current position
        if (n == 1) return 1;
        if (n == 2) return 2;

        if (arr[n] == 0) {
            arr[n] = totalWaysToStairTopDown(arr, n - 1) +
                     totalWaysToStairTopDown(arr, n - 2) +
                     totalWaysToStairTopDown(arr, n - 3)
        }
        return arr[n];
    }

    //DP bottom up
    public int totalWaysToStairBottomUp(int n){
        int[] arr = new int[n+1];
        arr[0] = 1;
        arr[1] = 1;
        arr[2] = 2;

        for (int i = 3; i < n+1; i++) {
            arr[i] = arr[i-1] + arr[i-2] + arr[1-3];
        }
        return arr[n];
    }

    //DP bottom up better
    public int totalWaysToStairBottomUpBetter(int n){
        int a = 1, b = 1, c = 2;
        int d;

        for (int i = 3; i < n+1; i++){
            d = a + b + c;
            a = b;
            b = c;
            c = d;
        }
        return c;
    }





    /*
    Min jumps to reach nth stair
    can move 1,2 or 3 steps at a time
    1 + min(T(n-1)+T(n-2)+T(n-3))
    base condition: n = 0 -> 0 jump
                    n = 1 || n = 2 || n = 3  -> 1 jump
     */
    //brute force, use recursion
    public int minJumpToStair (int n) {
        if (n == 0) return 0; // stay at current position
        if (n == 1) return 1;
        if (n == 2) return 1;
        if (n == 3) return 1;

        return 1 + Math.min(Math.min(minJumpToStair(n-1), minJumpToStair(n-2)), minJumpToStair(n-3));
    }

    //Top Down
    public int minJumpToStairTopDown (int n, int[] arr) {
        if (n == 0) return 0; // stay at current position
        if (n==1 || n==2 || n==3) return 1;

        if (arr[0] == 0) {
            arr[n] = 1+ Math.min(Math.min(minJumpToStairTopDown(n-1, arr), minJumpToStairTopDown(n-2, arr)),
                    minJumpToStairTopDown(n-3, arr));
        }
        return arr[n];
    }

    //Bottom up
    public int minJumpToStairBottomUp (int n) {
        int[] arr = new int[n+1];
        arr[0] = 0;
        arr[1] = 1;
        arr[2] = 1;
        arr[3] = 1;

        for (int i = 4; i < n+1; i++){
            arr[i] = 1+ Math.min(Math.min(arr[i-1], arr[i-2]), arr[i-3]);
        }
        return arr[n];
    }

    //Bottom up better
    public int minJumpToStairBottomUpBetter (int n) {
        int a = 0, b = 1, c = 1;
        int d;

        for (int i = 3; i<n+1; i++){
            d = 1 + Math.min(Math.min(a, b), c);
        }
        return c;
    }





    /*
    Longest Increasing subsequence (LIS)
    ex: [3 1 8 2 5] the LIS is [1 2 5], length 3
        [5 2 8 6 3 6 9 5] LIS is [2 3 6 9] length 4
     */
    public int LIS (int[] arr) {

    }




}
