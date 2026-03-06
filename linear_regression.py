class LinearRegressionScratch:
    column_count = 0
    row_count = 0

    # Read CSV file
    with open('fake_housing_data.csv', 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')

        try:
            header = next(csv_reader)
            column_count = len(header)

            if column_count > 0:
                row_count = 1
                row_count += sum(1 for _ in csv_reader)

        except StopIteration as e:
            # Handle empty file case
            print(e)
            pass

    x_train = [[0 for _ in range(column_count)] for _ in range(row_count - 1)]


    def __init__(self, df, learning_rate, __lambda):
        self.df = df
        self.learning_rate = learning_rate
        self.__lambda = __lambda

        # Linear Combination fx_wb_i = np.dot(x, w) + b




