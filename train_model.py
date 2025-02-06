import pickle
from model import ModelLibrary

if __name__ == '__main__':

    # データを読み込む
    with open('./training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    n_symbols = len(data)
        
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    # データをライブラリにセットする
    model.set_data(data)

    # データを準備する
    model.prepare_training_data()
    


    breakpoint()
        
