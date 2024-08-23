import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def train_and_predict(height_values, weight_values, new_height):
    reg = linear_model.LinearRegression()
    reg.fit(height_values, weight_values)
    return reg.predict([[new_height]])

if __name__ == "__main__":
    height = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
    weight = [16, 25, 36, 49, 64, 81, 100]
    
    plt.scatter(height, weight, color='black')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.show()

    predicted_weight = train_and_predict(height, weight, 12.0)
    print(f"Predicted weight for height 12.0 is {predicted_weight[0]}")
