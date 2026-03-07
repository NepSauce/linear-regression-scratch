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

            # Add the bias term to the linear model prediction
            linear_model_prediction += self.b
            # Calculate the error between the model prediction and the actual target value
            error = linear_model_prediction - yi_target_value

            for feature_i in range(feature_count):
                w_gradient_vector[feature_i] += error * xi_feature_vector[feature_i]

                if self.lambda_ > 0:
                    w_gradient_vector[feature_i] += (self.lambda_ / example_count) * self.w[feature_i]
            
            b_gradient += error
        
        for feature_i in range(feature_count):
            w_gradient_vector[feature_i] = w_gradient_vector[feature_i] / example_count
        
        b_gradient = b_gradient / example_count

        return w_gradient_vector, b_gradient

    def compute_cost(self):
        example_count = self.row_count
        feature_count = len(self.w)
        total_cost = 0.0

        for example_i in range(example_count):
            xi_feature_vector = self.x_train[example_i]
            yi_target_value = self.y_train[example_i]
            linear_model_prediction = 0.0

            for feature_i in range(feature_count):
                linear_model_prediction += self.w[feature_i] * xi_feature_vector[feature_i]

            linear_model_prediction += self.b
            error = linear_model_prediction - yi_target_value
            total_cost += error ** 2
        
        mean_squared_error = total_cost / (2 * example_count)

        if self.lambda_ > 0:
            l2_regularization_cost = (self.lambda_ / (2 * example_count)) * sum(wi ** 2 for wi in self.w)
            mean_squared_error += l2_regularization_cost
        
        return mean_squared_error
    

 
if __name__ == "__main__":
    lr = LinearRegressionScratch(0.01, 0, 0, 0.1)
    header = lr.load_csv("data.csv")

    compute_cost = lr.compute_cost()
    print(f"Initial Cost: {compute_cost}")

    compute_gradient = lr.compute_gradient()
    print(f"Weight Gradients: {compute_gradient[0]}, Bias Gradient: {compute_gradient[1]}")
    
    

    print(f"Column Count: {lr.column_count}, Row Count: {lr.row_count}")
