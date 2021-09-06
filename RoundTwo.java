public class RoundTwo {
    /*
    Search a 2D Matrix
    Binary Search
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        //exception
        if (matrix.length == 0 || matrix == null) return false;

        int rowIndex = matrix.length - 1;
        int columnIndex = matrix[0].length - 1;

        int start = 0;
        int end = rowIndex;

        //find possible row, if target is in the middle rows
        while (start + 1 < end){
            int mid = start + (start + end)/2;
            if (matrix[mid][0] == target) return true;
            if (matrix[mid][0] > target){
                start = mid;
            }
            if (target < matrix[mid][0]){
                end = mid;
            }
        }
        //check if target is in the first or last row
        if (matrix[start][0] <= target){
            rowIndex = start;
        }
        if (matrix[end][0] >= target){
            rowIndex = end;
        }

        //find column index
        start = 0;
        end = columnIndex;

        while (start + 1 < end) {
            int mid = start + (start + end) / 2;
            if (matrix[mid][0] == target) return true;
            if (matrix[mid][0] > target){
                start = mid;
            }
            if (target < matrix[mid][0]){
                end = mid;
            }
        }

        if (matrix[rowIndex][start] == target) return true;
        if (matrix[rowIndex][end] == target) return true;
        return false;
    }

    /*
    search 2D matrix
    think the matrix as a line of array
     */
    public boolean searchMatrixArr(int[][] matrix, int target) {
        int row = matrix.length;
        int column = matrix[0].length;

        int left = 0;
        int right = row * column - 1;

        while (left + 1 < right) {
            int mid = left + (left + right)/2;
            int mid_element = matrix[mid / column][mid % column];
            if (mid_element == target) return true;
            if (target < mid_element){
                right = mid;
            }
            if (target > mid_element){
                left = mid;
            }
        }
        if (matrix[right / column][right % column] == target ||
                matrix[left / column][left % column] == target ) {
            return true;
        }
        return false;
    }
}
