import pytest
from Lab2_LinearR import train_and_predict

def test_train_and_predict():
    height = [[1.0], [2.0], [3.0],[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    weight = [2,4,6,8,10,12,14,16,18,20]
    
    predicted_weight = train_and_predict(height, weight, 12.0)
    
    expected_min = 22
    expected_max = 26
    
    assert expected_min <= predicted_weight[0] <= expected_max, (
        f"Expected weight to be between {expected_min} and {expected_max}, "
        f"but got {predicted_weight[0]}"
    )

if __name__ == "__main__":
    pytest.main()
