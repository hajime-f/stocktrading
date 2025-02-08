import pickle
from model import ModelLibrary

if __name__ == '__main__':

    # データファイル名
    filename = 'training_data2.pkl'

    print(f'{filename} からデータを読み込んでいます。')
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    n_symbols = len(data)
    
    # モデルライブラリを初期化する
    model = ModelLibrary(n_symbols)
    
    # データをライブラリにセットする
    model.set_data(data)

    print('生データを準備しています。')
    raw_data = model.prepare_raw_data()

    print('学習データを準備しています。')
    X, Y = model.prepare_training_data(raw_data)

    print('モデルを評価しています。')
    max_clf = model.evaluate_model(X, Y)

    # from sklearn.ensemble import AdaBoostClassifier
    # max_clf = AdaBoostClassifier(n_estimators=180, random_state=1)

    print('モデルを検証しています。')
    trained_clf = model.validate_model(max_clf, X, Y)
    
    print('学習済みモデルを保存しています。')
    filename = model.save_model(trained_clf)

    print('学習済みモデルを復元しています。')
    clf = model.load_model(filename)

    print('モデルを再度検証しています。')
    trained_clf = model.validate_model(clf, X, Y)
    
    
    breakpoint()
    
    
    
