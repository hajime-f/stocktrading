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

## 仮想環境の作成

仮想環境を作ってください。前述のとおり、Python 3.12 の環境が必要です。

```
$ python -m venv env
```

## パッケージのインストール

仮想環境に入って ```make install``` してください。

```
$ source ./env/bin/activate
(env) $ make install
```

> [!CAUTION]
> インストール時にエラーが出る場合は、```requirements.txt``` から「wheel」と「playsound」をいったん削除し、再度 ```make install``` してみてください。
> その後、```pip install wheel playsound``` で wheel と playsound を別途インストールすれば、うまく動作するかもしれません。

## データベースの初期化

Yahoo! ファイナンスから株価データを取得し、データベースを初期化します。

```
(env) $ make init
```

## 予測モデル（日足）の学習

日足の値動きを予測するモデル（LSTM）を学習させます。分類性能を示すレポートが最後に出力されるので確認してください。後述するとおり、Recall が低く、Precision が高いことが理想です。

```
(env) $ make lstm
```

## .env ファイルの追加

プログラムディレクトリの直下に、下記のようにパスワード等を記録した .env ファイルを配置してください。なお、「APIPassword_production」は、証券会社から発行される API パスワード（本番用）です。

```:.env
APIPassword_production=XXXX
IPAddress=127.0.0.1
Port=18080
```

## データの収集

予測モデル（分足）を学習させるためのデータを集めてください。市場が開く 9:00 から 15:30 までの間、プログラムを動作させ続けてデータを収集します。

```
(env) $ make collect
```

15:30 になると PUSH 配信が止まるため、端末の表示も止まります。ほとぼりが冷めたくらいを見計らって、```Ctrl+C``` でプログラムを停止してください。\
ただし、```Ctrl+C``` を押してからプロンプトが返ってくるまで時間がかかります。辛抱強く待ってください。

データは ```./data``` 配下に生成されます。少なくとも５日分くらいのデータを集めましょう。

## 予測モデル（分足）の学習

分足の値動きを予測するモデルを学習させます。まず、```train_model.py``` の８行目に、集めたデータのファイル名を列挙してください。

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

学習が終了すると、```./model``` 配下にモデルのインスタンスが保存されます。

## 取引の開始

まず、```stocktrading.py``` の42行目に、モデルインスタンスのファイル名を記述してください。

```python
filename = os.path.join("./model/", 'model_20250215_233856.pkl')
```

次に、下記のコマンドを打つと取引が始まります。

```
(env) $ make
```

# API利用までの流れ

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

# Macを使った開発環境の構築方法

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

# 投資判断アルゴリズムの基本的な考え方

株取引で儲けるのに「株価が将来いくらになるか」を知る必要はありません。「株価がいまより上がるのかどうか」さえ分かれば十分です。こう割り切れば、問題のカテゴリーは「時系列予測」ではなく「二値分類」になります。

つまり、直近N分間の株価（および株価から計算される特徴量）に基づいて、M分以内に株価がK％上昇するか否かを分類する問題を解けばよいことになります。本プログラムは、この単純化（問題の読み替え）にしたがって株価の動向を「分類」し、投資判断を自動化しています。

具体的には、本プログラムは次の3ステップを踏んでいます。

## 直近10分間の株価から入力データを構成する

まず、始値、高値、安値、終値を1分ごとに計算します。つまり、kabuステーション経由でPUSH配信されるデータから、分足のチャートを構成します。なお、分類器で扱うことを考慮して、それぞれの値は正規化します。

次に、1分ごとの終値を使って各種のテクニカル指標（5分移動平均、25分移動平均、MACD、シグナル、ヒストグラム、ボリンジャーバンド上値・下値、RSI）を計算します。ここまでで、12種類の値が得られました。

そして、これらの値を10分間まとめて120次元のベクトルを構成します。これが分類器への入力データになります。

## 20分以内の株価から正解データを構成する

次に、各入力データに対応する20分以内の株価（終値）において、K％上昇しているポイントが存在するか否かを判定します。存在すればラベル1を与え、存在しなければラベル0を与えます。これが分類器に与える正解データになります。

## 分類器を学習させる

上記で構成した入力データ・正解データのペアを、scikit-learnに実装されている各分類器に与え、手当たり次第に評価していきます。最も高い正解率を出した分類器をチャンピオンモデルとして採用し、この性能を詳しく検証して実戦投入します。

## スタンス

株取引は、最先端技術を駆使した高頻度取引でプロがしのぎを削る世界ですから、上記の単純化でサクッと儲かるほど甘くないことは、もちろん承知しています。しかし、ローソク足チャートを使ったテクニカル分析だけで儲けを叩き出す個人投資家（デイトレーダー）がたくさんいることも事実で、彼ら／彼女らは「各種の特徴量から株価の上げ下げを分類している」と考えられます。

であれば、その投資判断をコンピュータで模擬し、儲けを出すことはできるはずです。\
このスタンスで本当に儲かるかどうかを検証したいと考えています。




