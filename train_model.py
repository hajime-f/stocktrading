import os
import pickle

from model import ModelLibrary


if __name__ == '__main__':

    # データファイル名
    filename_list = ['data_20250210_155746.pkl',]
    data = []
    n_symbols = 0
    
    for i, filename in enumerate(filename_list):
        print(f'{filename} からデータを読み込んでいます。')
        filename = os.path.join("./data/", filename)
        with open(filename, 'rb') as f:
            tmp = pickle.load(f)
            data += tmp
        n_symbols += len(tmp)
    
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    print('学習データを準備しています。')
    X, Y = model.prepare_training_data(data)

    print('モデルを評価しています。')
    best_clf = model.evaluate_model(X, Y)

    print('モデルを検証しています。')
    clf = model.validate_model(best_clf, X, Y)
    
    print('学習済みモデルを保存しています。')
    filename = model.save_model(clf)

    print('学習済みモデルを復元しています。')
    clf = model.load_model(filename)

    print('モデルを再度検証しています。')
    trained_clf = model.validate_model(clf, X, Y)
    
