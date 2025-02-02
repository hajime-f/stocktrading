import pandas as pd


class Train:

    def __init__(self, n_symbols):

        self.columns = ['DateTime', 'Price']
        self.data = []
        for i in range(n_symbols):
            self.data.append(pd.DataFrame(index = [], columns = self.columns))
            self.data[i].set_index('DateTime')


    def append_data(self, formatted_datetime, price, index):

        new_data = pd.DataFrame([[formatted_datetime, price]], columns = self.columns)
        self.data = pd.concat([self.data[index], new_data])


    def train_model(self):
        pass

