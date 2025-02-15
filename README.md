# kabu station API を使った日本株の自動売買プログラム

[kabuステーションAPI](https://kabucom.github.io/kabusapi/ptal/) を使って、日本株を自動的に売買するプログラムです。

> [!CAUTION]
> 本プログラムを使用することによって被った損害等について、制作者は一切の責任を負いません。
> 投資はご自身の判断と責任のもとで行ってください。

## 使い方

### .env ファイルの追加

stocklib ディレクトリの直下に、下記のようにパスワード等を記録した .env ファイルを配置してください。

```:.env
APIPassword_production=XXXX
OrderPassword=YYYY
IPAddress=127.0.0.1
Port=18080
```

### 仮想環境の作成

仮想環境を作ってください。

```
$ python -m venv env
```

### パッケージのインストール

仮想環境に入って ```make install``` してください。

```
$ source ./env/bin/activate
(env) $ make install
```
### データの収集

下記のコマンドを打って、予測モデルを学習させるためのデータを集めてください。

```
(env) $ make data
```

データは ```./data``` 配下に生成されます。\
少なくとも３日分くらいのデータが必要です。

### 予測モデルの学習

予測モデルを学習させます。\
まず、```train_model.py``` の８行目に、集めたデータのファイル名を列挙してください。

```python
filename_list = ['data_20250206_201528.pkl', 'data_20250207_201857.pkl', ... , 'data_20250212_192845.pkl']
```

次に、下記のコマンドを打ってモデルを学習させてください。

```
(env) $ make train
```

