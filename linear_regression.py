import csv

class LinearRegressionScratch:
    def __init__(self, learning_rate, w, b, lambda_):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.x_train = []
        self.y_train = []
        self.w = w
        self.b = b
        self.column_count = 0
        self.row_count = 0

    def load_csv(self, file_path):
        try:
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file, delimiter=',')
                header = next(csv_reader)
                self.column_count = len(header)

                for row in csv_reader:
                    self.x_train.append([float(value) for value in row[:-1]])
                    self.y_train.append(float(row[-1]))
                    self.row_count += 1
            
            n_features = self.column_count - 1
            self.w = [0.0] * n_features

            return header
        
        except Exception as e:
            print(f"Error loading CSV: {e}")

            return None
        
    def compute_gradient(self):
        # Assign total number of examples and features
        example_count = self.row_count
        feature_count = len(self.w)
        # Initialize gradients for weights and bias
        w_gradient_vector = [0.0] * feature_count
        b_gradient = 0.0

        # Loop through each training example to compute the prediction, error, and accumulate gradients
        for example_i in range(example_count):
            # Extract the feature vector and target value for the current example
            xi_feature_vector = self.x_train[example_i]
            yi_target_value = self.y_train[example_i]
            # Initialize model prediction as 0.0
            linear_model_prediction = 0.0

            # Compute the linear model prediction by summing the product of weights and features
            for feature_i in range(feature_count):
                linear_model_prediction += self.w[feature_i] * xi_feature_vector[feature_i]

            # Add the bias term to the model prediction
            linear_model_prediction += self.b
            # Calculate the error between the model prediction and the actual target value
            error = linear_model_prediction - yi_target_value

            # Update the gradients for weights by multiplying the error with the corresponding feature value and accumulating it
            for feature_i in range(feature_count):
                w_gradient_vector[feature_i] += error * xi_feature_vector[feature_i]

                # If regularization is applied, add the regularization term to the weight gradient
                if self.lambda_ > 0:
                    w_gradient_vector[feature_i] += (self.lambda_ / example_count) * self.w[feature_i]
            
            # Update the bias gradient by adding the error
            b_gradient += error
        
        # Average the gradients by dividing by the number of examples
        for feature_i in range(feature_count):
            w_gradient_vector[feature_i] = w_gradient_vector[feature_i] / example_count
        
        # Average the bias gradient by dividing by the number of examples
        b_gradient = b_gradient / example_count

        return w_gradient_vector, b_gradient

    def compute_cost(self):
        # Extract the total number of examples and features from the training data
        example_count = self.row_count
        feature_count = len(self.w)
        # Initialize the total cost to 0.0
        total_squared_cost = 0.0

        # Loop through each training example to compute the prediction and accumulate the squared cost
        for example_i in range(example_count):
            # Extract the feature vector and target value for the current example
            xi_feature_vector = self.x_train[example_i]
            yi_target_value = self.y_train[example_i]
            # Initialize model prediction as 0.0
            linear_model_prediction = 0.0

            # Compute the linear model prediction by summing the product of weights and features
            for feature_i in range(feature_count):
                linear_model_prediction += self.w[feature_i] * xi_feature_vector[feature_i]

            # Add the bias term to the model prediction
            linear_model_prediction += self.b
            # Calculate the error between the model prediction and the actual target value
            error = linear_model_prediction - yi_target_value
            # Accumulate the squared cost by adding the square of the error
            total_squared_cost += error ** 2
        
        # Average the total squared cost by dividing by twice the number of examples
        average_squared_cost = total_squared_cost / (2 * example_count)

        # If regularization is applied, compute the regularization cost and add it to the average squared cost
        if self.lambda_ > 0:
            sum_squared_weights = 0.0

            for feature_i in range(feature_count):
                sum_squared_weights += self.w[feature_i] ** 2
            
            average_squared_cost += (self.lambda_ / (2 * example_count)) * sum_squared_weights

        return average_squared_cost
    
    def gradient_descent_step(self):
        # Compute the gradients for weights and bias
        w_gradient_vector, b_gradient = self.compute_gradient()

        # Update the weights by subtracting the product of learning rate and weight gradients from the current weights
        for feature_i in range(len(self.w)):
            self.w[feature_i] -= self.learning_rate * w_gradient_vector[feature_i]
        
        # Update the bias by subtracting the product of learning rate and bias gradient from the current bias
        self.b -= self.learning_rate * b_gradient

            

if __name__ == "__main__":
    lr = LinearRegressionScratch(0.01, 0, 0, 0.1)
    header = lr.load_csv("data.csv")

    compute_cost = lr.compute_cost()
    print(f"Initial Cost: {compute_cost}")

    compute_gradient = lr.compute_gradient()
    print(f"Weight Gradients: {compute_gradient[0]}, Bias Gradient: {compute_gradient[1]}")
    
    

    print(f"Column Count: {lr.column_count}, Row Count: {lr.row_count}")
