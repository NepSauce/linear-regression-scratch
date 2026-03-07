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

    def get_csv_header(self, file_path):
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            header = next(csv_reader)

        return header
    
    def get_csv_dimensions(self, header, file_path):
        self.column_count = len(header)
        self.row_count = sum(1 for _ in open(file_path)) - 1
    
if __name__ == "__main__":
    lr = LinearRegressionScratch(0.01, 0, 0, 0.1)
    header = lr.get_csv_header("data.csv")
    lr.get_csv_dimensions(header, "data.csv")
    

    print(f"Column Count: {lr.column_count}, Row Count: {lr.row_count}")




    # print(f"Column Count: {self.column_count}")
    # print(f"Row Count: {self.row_count}")

            # for each_row in rows:
            #     self.x_train.append([float(value) for value in each_row[:-1]])
            #     self.y_train.append(float(each_row[-1]))


        

        # Linear Combination fx_wb_i = np.dot(x, w) + b
