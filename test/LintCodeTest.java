import org.junit.Test;

import static org.junit.Assert.*;

public class LintCodeTest {

    @Test
    public void twoSumTest() {
        LintCode testClass = new LintCode();
        int[] numbers = new int[] {0, 4, 3, 11, 10,2, 7 ,9};
        int target = 5;
        int[] result = testClass.twoSum(numbers, target);
        int[] expected = new int[] {2,5};
        assertArrayEquals(expected, result);
    }

    @Test
    public void mergeSortedArrayTest() {
        LintCode testClass = new LintCode();
        int[] A = new int[] {1, 3, 5};
        int[] B = new int[] {2, 4};
        int m = 3, n = 2;
        testClass.mergeSortedArray(A, m, B, n);
        int[] expected = new int[] {1, 2, 3, 4, 5};
        assertArrayEquals(expected, A);
    }
}