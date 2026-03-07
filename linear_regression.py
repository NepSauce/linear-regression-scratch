import csv

class LinearRegressionScratch:
    def __init__(self, learning_rate, w, b, lambda_):
        self.EULERS_CONSTANT = 2.718281828459
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.x_train = []
        self.y_train = []
        self.w = w
        self.b = b
        self.column_count = 0
        self.row_count = 0

    def read_csv(self, file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            header = next(csv_reader)

        return header
    
    def csv_dimensions(self, csv_header):
        rows = list(csv_header)
        self.column_count = len(csv_header)
        self.row_count = len(rows)

        return self.column_count, self.row_count
    
    def sigmoid(self, z):
        return 1 / (1 + self.EULERS_CONSTANT ** -z)
    
if __name__ == "__main__":
    lr = LinearRegressionScratch(0.01, 0, 0, 0.1)
    header = lr.read_csv("data.csv")
    column_count, row_count = lr.csv_dimensions(header)

    print(f"Column Count: {column_count}, Row Count: {row_count}")




    # print(f"Column Count: {self.column_count}")
    # print(f"Row Count: {self.row_count}")

            # for each_row in rows:
            #     self.x_train.append([float(value) for value in each_row[:-1]])
            #     self.y_train.append(float(each_row[-1]))


        

        # Linear Combination fx_wb_i = np.dot(x, w) + b
