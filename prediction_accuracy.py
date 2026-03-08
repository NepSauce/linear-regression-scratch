import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PredictionAccuracy:
    def __init__(self, learning_rate, w, b, lambda_):
        data = pd.read_csv('data.csv')
        X = data[['size_sqft', 'bedrooms', 'age']].values
        y = data['price'].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        model = nn.Linear(3, 1)
        criterion = nn.MSELoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
