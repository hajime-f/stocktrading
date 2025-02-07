import pickle
from model import ModelLibrary

if __name__ == '__main__':

    filename = 'training_data2.pkl'
    
    # データを読み込む
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    n_symbols = len(data)
        
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    # データをライブラリにセットする
    model.set_data(data)

    # データを準備する
    raw_data = model.prepare_raw_data()
    
    
    
