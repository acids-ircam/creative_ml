import pytest
from assignment import multiply_by_2, fibonacci

def test_multiply():
    assert multiply_by_2(2) == 4
    assert multiply_by_2(5) == 10
    assert multiply_by_2(-3) == -6
    assert multiply_by_2(0) == 0
    assert multiply_by_2(1.1) == 2.2
