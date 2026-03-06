import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionScratch:
    df = pd.read_csv("fake_housing_data.csv")

    def __init__(self, df, learning_rate, __lambda):
        self.df = df
        self.learning_rate = learning_rate
        self.__lambda = __lambda

        # Linear Combination f_x_wb_i = np.dot(x, w) + b



