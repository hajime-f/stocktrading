import pickle
from model import ModelLibrary

if __name__ == '__main__':

    # データファイル名
    filename_list = ['data_20250210_155746.pkl',]
    data = []
    n_symbols = 0
    
    for i, filename in enumerate(filename_list):
        print(f'{filename} からデータを読み込んでいます。')
        with open(filename, 'rb') as f:
            data.append(pickle.load(f))
        n_symbols += len(data[i])
    
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    # データをライブラリにセットする
    model.set_data(data)

    print('生データを準備しています。')
    XY = model.prepare_dataframe_list()

    print('学習データを準備しています。')
    X, Y = model.prepare_training_data(XY)

    print('モデルを評価しています。')
    max_clf = model.evaluate_model(X, Y)

    print('モデルを検証しています。')
    trained_clf = model.validate_model(max_clf, X, Y)
    
    print('学習済みモデルを保存しています。')
    filename = model.save_model(trained_clf)

    print('学習済みモデルを復元しています。')
    clf = model.load_model(filename)

    print('モデルを再度検証しています。')
    trained_clf = model.validate_model(clf, X, Y)
    
