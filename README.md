# 機械学習を応用した日本株の自動売買プログラム

> [!CAUTION]
> 本プログラムを使用することによって被った損害等について、制作者は一切の責任を負いません。
> 投資はご自身の判断と責任のもとで行ってください。

# 動作環境

- Python 3.12\
2025年3月現在において、TensorFlow が Python 3.13 に対応していないため、Python 3.12 を使用します。3.13 では動作しませんので、ご注意ください。
- [kabuステーションAPI](https://kabucom.github.io/kabusapi/ptal/) \
[三菱UFJ eスマート証券](https://kabu.com/)が提供する株取引専用の API を使用します。証券口座の開設が別途必要です。

# 動作方法

## 1. 仮想環境の作成

仮想環境を作ってください。前述のとおり、Python 3.12 の環境が必要です。

```
$ python -m venv env
```

## 2. パッケージのインストール

仮想環境に入って ```make install``` してください。

```
$ source ./env/bin/activate
(env) $ make install
```

> [!TIP]
> インストール時にエラーが出る場合は、```requirements.txt``` から「wheel」と「playsound」をいったん削除し、再度 ```make install``` してみてください。
> その後、```pip install wheel playsound``` でこれらを別途インストールすれば、うまく動作するかもしれません。

## 3. .env ファイルの追加

プログラムディレクトリの直下に、下記のようにパスワード等を記録した .env ファイルを配置してください。なお、「APIPassword_production」は、証券会社から発行される API パスワード（本番用）です。また、「BaseDir」は、本プログラムが格納されている「stocktrading」ディレクトリへの絶対パスです。

```:.env
APIPassword_production=XXXX
IPAddress=127.0.0.1
Port=18080
BaseDir=/path/to/dir/stocktrading
```

## 4. データベースの初期化

Yahoo! ファイナンスから株価データを取得し、データベースを初期化します。

```
(env) $ make init
```

## 5. 予測モデル（日足）の学習

日足の値動きを予測するモデルを学習させます。

```
(env) $ make update
```

学習が終了すると、```./model``` 配下に ```model_swingtrade_（保存した日時）_（パラメータ）.keras``` という名前でモデルのインスタンスが保存されます。

## 6. データの収集

予測モデル（分足）を学習させるためのデータを集めてください。9:00 から 15:30 までの間、プログラムを実行し続けてデータを収集します。

```
(env) $ make collect
```

15:30 になると PUSH 配信が止まるため、端末の表示も止まります。ほとぼりが冷めたくらいを見計らって、```Ctrl+C``` でプログラムを停止してください。ただし、```Ctrl+C``` を押してからプロンプトが返ってくるまで時間がかかります。辛抱強く待ってください。

データは ```./data``` 配下に生成されます。３〜４日分くらいのデータを集めましょう。

## 7. 予測モデル（分足）の学習

分足の値動きを予測するモデル（LSTM）を学習させます。まず、```train_model.py``` の８行目に、集めたデータのファイル名を列挙してください。

```python
filename_list = [
	'data_20250206_201528.pkl', 
	'data_20250207_201857.pkl', ... , 
	'data_20250212_192845.pkl'
]
```

次に、下記のコマンドを打ってモデルを学習させてください。

```
(env) $ make train
```

学習が終了すると、```./model``` 配下に ```model_daytrade_（保存した日時）.keras``` という名前でモデルのインスタンスが保存されます。

## 8. 日足の予測

日足の値動きを予測します。下記のコマンドを打って予測を実行しましょう。

```python
(env) $ make predict
```

> [!TIP]
> 「4. データベースの初期化」「5. 予測モデル（日足）の学習」「8. 日足の予測」は cron に登録しておいて、毎日実行するようにしておくと便利です。私はそれぞれ 18:00, 20:00, 0:00 に実行するようにしています。

## 9. 取引の開始

まず、```stocktrading.py``` の43行目に、モデルインスタンス（分足）のファイル名を記述してください。

```python
filename = os.path.join("./model/", 'model_daytrade_20250315_233856.keras')
```

次に、下記のコマンドを打つと取引が始まります。

```
(env) $ make
```

# 投資判断アルゴリズムの概要

## ２つの予測モデル

本プログラムは、次の2つの予測モデルを使います。

- 翌営業日の終値が、当日の終値から上がるか否かを予測するモデル（日足モデル）
- 当日のある時点での株価を基準にして、そこから20分以内に上がるか否かを予測するモデル（分足モデル）

いずれも、時系列データに基づいて上がる（クラス1）か上がらない（クラス0）かを予測する深層学習モデルです。

深層学習モデルといえども、カオティックな株価の値動きを正確に予測することはできません。
そこで、「予測は基本的に当たらない」という事実を受け入れて、より現実的な方針を採ることにします。

それは、「取りこぼしは多くてもいいから、確実に上がる銘柄とタイミングだけをすくい上げる」という方針です。
「Recall が低いことを許容し、Precision を上げる」と言い換えてもよいでしょう。

例えば、日足モデルのラベル１予測精度は、Recall：0.06、Precision：0.82 程度です（2025年3月現在）。
つまり、ラベル１データの発見率は6％、発見したデータが真にラベル１である確率は82％です。

通常であれば、このように「見逃し」の多い（6％しか発見できない！）偏ったモデルは実用に適しません。
しかし、「株取引で儲ける」という目的であれば、十分に実用可能です。
すくい上げたチャンスがわずかであっても、そのうちの82％が「本当のチャンス」なので、これを確実に掴めばよいからです。

このように、本プログラムは、不確かな予測でも儲かるアルゴリズムを追求しています。
具体的には、下記のとおりに株取引を実行します。

## 1. 日足モデルで「値上がりしそうな銘柄」を絞り込む

東証に上場している約 4000 銘柄から、

- ETF や REIT などの非企業の銘柄
- 上場から間がなくデータが不十分な銘柄
- 出来高が少なく取引に向かない銘柄

を除外すると、約 2000 銘柄になります。このなかから「当日の終値を基準にして、翌営業日の終値が 0.5 ％超上がる銘柄」を、日足モデルで発見します。
具体的には、次の4ステップで 2000 銘柄を 50 銘柄に絞り込みます。

### 1-1. 直近20日分の株価から入力データを構成する

Yahoo! ファイナンスから取得した株価データを利用して、1日あたり 21 次元のデータを構成します。これを20日分束ねた 21x20 の行列がモデルの入力となります。

### 1-2. 翌営業日の株価から正解データを構成する

翌営業日の終値が当日の終値から 0.5 ％上昇していたら、クラス1のラベルを付与します。

### 1-3. 日足モデルを学習させる

21x20 の行列とそれに対応するラベルのペアを与えて、日足モデルを学習させます。なお、モデルは6層からなる DNN で、2層目に双方向 RNN を利用しています。

```python:rnn.py
def DNN_compile(self, array):
    model = Sequential()

    model.add(InputLayer(shape=(array.shape[1], array.shape[2])))
    model.add(Bidirectional(SimpleRNN(200)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
```

### 1-4. 当日までのデータを使って翌営業日の値上がりを予測する

モデルが出力する予測値（0〜1の値をとる）を、ラベル0または1に振り分けるための閾値を調整することで、2000 銘柄から 50 銘柄に絞り込みます。
なお、絞り込む数を 50 にするのは、kabu ステーション API の利用制限（取引対象となる「登録銘柄」は 50 銘柄以下）があるためです。

Precision を高く保つために閾値は高めに固定しておきたいのですが、高くしすぎると過剰に絞り込むことになる（数個の銘柄しか検出できない）ため、「50」の絞り込みを前提として閾値を動的に設定します。

その結果、閾値は 0.55〜0.65 に設定されます（日によって変わります）。当然、Precision は下がる（実際には値上がりしない銘柄が多く混入する）ことになりますが、ランダムに選抜するよりは良好な絞り込みが実現できます。

## 2. 分足モデルを使ってデイトレードを行う

日足モデルを使って絞り込んだ 50 銘柄を対象にして、当日にデイトレードを行います。


### 2-1. 直近10分間の株価から入力データを構成する

事前準備として、まずは過去のデータから始値、高値、安値、終値を1分ごとに計算します。つまり、kabu ステーション経由で PUSH 配信されるデータから、分足のチャートを構成します。

次に、1分ごとの終値を使って各種のテクニカル指標（5分移動平均、25分移動平均、MACD、シグナル、ヒストグラム、ボリンジャーバンド上値・下値、RSI）を計算します。ここまでで、12種類の値が特徴として得られました。

そして、これらの値を10分間まとめて 12x10 の行列を構成します。これがモデルへの入力データになります。

### 2-2. 20分以内の株価から正解データを構成する

次に、各入力データに対応する20分以内の株価（終値）において、0.5 ％上昇しているポイントが存在するか否かを判定します。存在すればラベル1を与え、存在しなければラベル0を与えます。これがモデルに与える正解データになります。

### 2-3. 分足モデルを学習させる

上記で構成した入力データ・正解データのペアをモデルに与え、学習させます。なお、モデルは6層からなる DNN で、2層目に LSTM を利用しています。

```python:model.py
def compile_model(self, shape1, shape2):
    model = Sequential()

    model.add(InputLayer(shape=(shape1, shape2)))
    model.add(LSTM(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
```

### 2-4. 

## スタンス

株取引は、最先端技術を駆使した高頻度取引でプロがしのぎを削る世界ですから、上記の方針でサクッと儲かるほど甘くないことは、もちろん承知しています。しかし、ローソク足チャートを使ったテクニカル分析だけで儲けを叩き出す個人投資家（デイトレーダー）がたくさんいることも事実で、彼ら／彼女らは「各種の特徴量から株価の上げ下げを予測している」と考えられます。

であれば、その投資判断をコンピュータで模擬し、儲けを出すことはできるはずです。\
このスタンスで本当に儲かるかどうかを検証したいと考えています。

# 【番外１】API利用までの流れ

kabuステーションAPIを利用可能にするまでには、越えなければならない山がいくつかあります。\
以下では、その乗り越え方を説明します。

## 三菱UFJ eスマート証券に証券口座を開設する

[トップページ](https://kabu.com/)から「口座開設」を選択し、ガイダンスに沿って必要事項を入力していけば口座が開設できます。\
なお、口座は「特定口座 源泉徴収あり」にしておきましょう。「源泉徴収なし」にすると確定申告が面倒なので。

## kabuステーションをインストールする

Windowsマシンにkabuステーションをインストールします。\
なお、Windowsマシンは実機である必要はありません。私は[Parallels](https://www.parallels.com/jp/)上で動作する仮想Windowsマシンでkabuステーションを動作させています。

## APIの利用設定を行う

[このガイダンス](https://kabucom.github.io/kabusapi/ptal/howto.html)に沿って利用設定を進めます。\
しかし、kabuステーションAPIの「状態」が「利用可」にならず、途中でつまづくと思います。

これは、APIを利用するためには「Professionalプラン」である必要があるのですが、現段階ではその条件を満たしておらず、「通常プラン」でしかないからです。

## Professionalプラン移行のための条件を満たす

[このページ](https://kabu.com/tool/kabustation/default.html)の「ご利用プランについて」の記載によれば、Professionalプラン移行には次の2つの条件を満たす必要があります。

1. 信用取引口座または、先物オプション取引口座開設済み
2. 前々々月～前営業日で当社全取引における約定回数が1回以上ある

1については信用取引の口座を開設するだけなので簡単なのですが、問題は2です。とにかくなんでもいいので株取引を少なくとも１回約定させ、取引実績を作る必要があります。

私は、銘柄1475（iシェアーズ・コア TOPIX ETF）を買ってすぐ売却しました。この銘柄は値段が安く、10株単位で取引できるので、数千円あれば取引実績を作ることができます。

## APIの利用設定を行う（再）

約定の翌営業日に、APIの利用設定ができるようになっているはずです。

## APIシステム設定を行う

kabuステーションを起動し、[このページ](https://kabucom.github.io/kabusapi/ptal/howto.html)のガイダンスに沿って「APIシステム設定」を進めます。このガイダンスに記載のとおり、kabuステーションの画面右上のアイコンが緑色になっていることを確認しましょう。

# 【番外２】Macを使った環境の構築方法

「kabuステーションAPI」は、[「kabuステーション」](https://kabu.com/kabustation/default.html)と呼ばれる株取引専用ソフトウェアを経由しなければ使用できません。つまり、原則としてkabuステーションが動作するPC上でプログラムを実行することが求められ、このとき「localhost」をホストに指定して各エンドポイントにアクセスすることになります。

しかし、残念ながら、kabuステーションはWindows版しか提供されていません。\
そのため、Macでプログラムを開発・実行するためには、テクニカルな工夫が必要になります。

以下では、Macにおける開発環境の構築方法について説明します。

## nginx をインストールする

kabuステーションが動作するWindowsマシンに、[nginx](https://nginx.org/en/)をインストールしてください。

そして、その設定を下記のように書き換えます。
```
worker_processes  1;
events {
    worker_connections  1024;
}
http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    keepalive_timeout  65;

    map $http_upgrade $connection_upgrade { 
    default upgrade;
    ''      close;
    } 

    server {
        listen       80;
        server_name  localhost;

        proxy_http_version 1.1;
        proxy_set_header Host localhost;
        proxy_set_header Upgrade $http_upgrade; 
        proxy_set_header Connection $connection_upgrade;

        location / {
            proxy_pass   http://127.0.0.1:18080/;
        }
    }
}
```
上記のように設定ファイルを書き換えたら、nginxを起動します。

## WindowsマシンのローカルIPアドレスを調べる

WindowsマシンのローカルIPアドレスを調べてください。コマンドプロンプトで```$ ipconfig /all```と打てば表示されます。

## .envファイルを書き換える

本プログラムのディレクトリ直下にある```.env```を下記のように書き換えてください。

```
APIPassword_production=XXXXX ← 証券会社から付与されたAPIパスワード
IPAddress=192.168.0.ZZZ ← 先ほど調べたWindowsマシンのローカルIPアドレス
Port=80
```

## プログラムを実行する

とりあえず、```$ make``` してみましょう。うまくいけば、買付余力が表示されるはずです。

## その他

kabuステーションは毎日再起動する必要があることに注意しましょう。\
APIキーは毎日リフレッシュ（更新）する必要があるのですが、kabuステーションを再起動しなければ、これができないようになっています。

なぜこのクソみたいな仕様になっているのかは分かりません。




