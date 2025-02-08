import pickle
from model import ModelLibrary

if __name__ == '__main__':

    # データファイル名
    filename = 'training_data2.pkl'

    print(f'{filename} からデータを読み込んでいます。')
    
    # データを読み込む
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    n_symbols = len(data)
        
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    # データをライブラリにセットする
    model.set_data(data)

    print('生データを準備しています。')
    
    # 生データを準備する
    raw_data = model.prepare_raw_data()

    # columns = ["close", "volume", "MA5", "MA25", "MACD", "SIGNAL", "HISTOGRAM", "Upper", "Lower", "RSI"]
    # raw_data[0][columns].iloc[0:10, :]

    print('学習データを準備しています。')
    X, Y = model.prepare_training_data(raw_data)

    print('モデルを学習しています。')
    model.train_model(X, Y)
    
    breakpoint()
    
    
    
