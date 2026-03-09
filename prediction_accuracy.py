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


    def predict(self, x_vector):
        # x_vector: list or iterable of 3 feature values
        if not isinstance(x_vector, torch.Tensor):
            new_house = torch.tensor([x_vector], dtype=torch.float32)
        else:
            new_house = x_vector.view(1, -1).to(torch.float32)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(new_house)

        return float(prediction.item())

    def compute_prediction_accuracy(self, x_vector=None):
        if x_vector is None:
            x_vector = [2500, 4, 5]

        pred = self.predict(x_vector)
        print(f'Predicted price: {pred}')
        return pred