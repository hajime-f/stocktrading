import pandas as pd


class Train:

    def __init__(self, n_symbols):

        self.n_symbols = n_symbols
        self.data = []
        for i in range(n_symbols):
            self.data.append([])


    def append_data(self, formatted_datetime, price, index):
        
        new_data = {'DateTime': formatted_datetime, 'Price': price}
        self.data[index].append(new_data)
        

    def train_model(self):

        columns = ['DateTime', 'Price']
        training_data = []
        
        for i in range(self.n_symbols):
            training_data.append(pd.DataFrame(index = [], columns = columns))
            training_data[i] = pd.to_datetime(training_data[i]['DateTime'])
            training_data[i] = pd.concat([training_data[i], pd.DataFrame(self.data[i], columns = columns)], ignore_index = True)
            training_data[i].drop_duplicates(subset = 'DateTime', keep = 'first', inplace = True)
            training_data[i] = training_data[i].set_index('DateTime')
            training_data[i] = training_data[i]['Price']
        
        breakpoint()

