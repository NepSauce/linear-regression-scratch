import csv

class LinearRegressionScratch:
    column_count = 0

    def __init__(self, learning_rate, __lambda):
        self.learning_rate = learning_rate
        self.__lambda = __lambda

        # Read CSV file
        with open('data.csv', 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')

            try:
                header = next(csv_reader)
                column_count = len(header)

                if column_count > 0:
                    row_count = 0
                    row_count += sum(1 for _ in csv_reader)

            except StopIteration as e:
                # Handle empty file case
                print(e)
                pass


            print(f"Column Count: {column_count}")
            print(f"Row Count: {row_count}")

            x_train = [[0 for _ in range(column_count - 2)] for _ in range(row_count - 1)]
            y_train = [0 for _ in range(row_count - 1)]

            for each_row in csv_reader:
                x_train.append(each_row[:-1])
                y_train.append(each_row[-1])
                
        
        print(f"Column Count: {column_count}")
        print(f"Row Count: {row_count}")

        x_train = [[0 for _ in range(column_count)] for _ in range(row_count - 1)]

        # Linear Combination fx_wb_i = np.dot(x, w) + b
