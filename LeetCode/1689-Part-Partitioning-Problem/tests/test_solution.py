import pytest
from solution import Solution


class TestPartitionToDeciBinary:
    def test_example_1(self):
        """n = "32" -> 3 deci-binary numbers: 11 + 11 + 10 = 32"""
        assert Solution().partiton("32") == 3

    def test_example_2(self):
        """n = "82734" -> 8 deci-binary numbers (max digit is 8)"""
        assert Solution().partiton("82734") == 8

    def test_single_digit_1(self):
        """n = "1" -> already deci-binary"""
        assert Solution().partiton("1") == 1

    def test_single_digit_9(self):
        """n = "9" -> 9 deci-binary numbers (1+1+1+1+1+1+1+1+1)"""
        assert Solution().partiton("9") == 9

    def test_all_ones(self):
        """n = "111" -> already deci-binary"""
        assert Solution().partiton("111") == 1

    def test_mixed_digits(self):
        """n = "123" -> 3 (max digit is 3)"""
        assert Solution().partiton("123") == 3

    def test_large_number(self):
        """n = "27346209830709182346" -> 9 (max digit is 9)"""
        assert Solution().partiton("27346209830709182346") == 9
