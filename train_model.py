import pickle
from library import StockLibrary

if __name__ == '__main__':

    # ライブラリを初期化する
    lib = StockLibrary()

    # データを読み込む
    with open('./training_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # データをライブラリにセットする
    lib.set_data(data, len(data))

    # データを整える
    lib.prepare_training_data()
    


    breakpoint()
        
