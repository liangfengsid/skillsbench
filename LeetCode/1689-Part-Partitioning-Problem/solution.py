class Solution:
    def partiton(self, n: str) -> int:
        """
        A deci-binary number consists of only digits 0 and 1.
        The minimum number of deci-binary numbers needed to sum to n
        equals the maximum digit in n, since each deci-binary number
        can contribute at most 1 to any digit position.
        """
        return int(max(n))
