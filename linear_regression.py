import csv

class LinearRegressionScratch:
    def __init__(self, learning_rate, w, b, lambda_):
        self.learning_rate = learning_rate
        self.__lambda = lambda_
        self.x_train = []
        self.y_train = []
        self.w = w
        self.b = b

        # Read CSV file
        with open('data.csv', 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')

            header = next(csv_reader)
            self.column_count = len(header)

            rows = list(csv_reader)
            self.row_count = len(rows)


            print(f"Column Count: {self.column_count}")
            print(f"Row Count: {self.row_count}")

            for each_row in rows:
                self.x_train.append([float(value) for value in each_row[:-1]])
                self.y_train.append(float(each_row[-1]))
                
        
        print(f"Column Count: {self.column_count}")
        print(f"Row Count: {self.row_count}")

        

        # Linear Combination fx_wb_i = np.dot(x, w) + b
