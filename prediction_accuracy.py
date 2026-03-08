import pandas as pd
import torch
import torch.nn as nn

class PredictionAccuracy:
    def __init__(self, learning_rate):
        data = pd.read_csv('data.csv')
        X = data[['size_sqft', 'bedrooms', 'age']].values
        y = data['price'].values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        # store model
        self.model = nn.Linear(3, 1)
        # loss function
        self.criterion = nn.MSELoss()
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)


    def train(self, epochs=1000):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(self.X)
            loss = self.criterion(predictions, self.y)
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')


    def compute_prediction_accuracy(self):
        new_house = torch.tensor([[2500, 4, 5]], dtype=torch.float32)
        prediction = self.model(new_house)

        print(f'Predicted price: {prediction.item()}')