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

Yahoo! ファイナンスから株価データを取得し、データベースを初期化します。なお、下記コマンドは新しいデータ（当日分の株価）の取り込みにも使うので、更新のために毎日実行する必要があります。

```
(env) $ make init
```

## 5. 予測モデルの学習

日足の値動きを予測するモデルを学習させます。

```
(env) $ make update
```

学習が終了すると、```./model``` 配下に ```model_swingtrade_（保存した日時）_（パラメータ）.keras``` という名前でモデルのインスタンスが保存されます。また、データベースの「Models」テーブルに、モデルのファイル名が記録されます。

## 6. 日足の予測

日足の値動きを予測します。下記のコマンドを打って予測を実行しましょう。

```python
(env) $ make predict
```

予測が終了すると、データベースの「Target」テーブルに、翌日の取引対象となる銘柄の情報が記録されます。

> [!TIP]
> 「4. データベースの初期化」「5. 予測モデル（日足）の学習」「6. 日足の予測」は cron に登録しておいて、毎日実行するようにしておくと便利です。私はそれぞれ 18:00, 20:00, 0:00 に実行するようにしています。

## 7. 取引の開始

下記のコマンドを打つと取引が始まります。

```
(env) $ make
```

# 投資判断アルゴリズムの概要

## 予測モデル

本プログラムは「翌営業日の終値が当日の終値から上がるか否かを予測するモデル」を使います。
これは、時系列データに基づいて上がる（クラス1）か上がらない（クラス0）かを予測する深層学習モデルです。

深層学習モデルといえども、カオティックな株価の値動きを正確に予測することはできません。
そこで、「予測は基本的に当たらない」という事実を受け入れて、より現実的な方針を採ることにします。

それは、「取りこぼしは多くてもいいから、確実に上がる銘柄だけをすくい上げる」という方針です。
「Recall が低いことを許容し、Precision を上げる」と言い換えてもよいでしょう。

実際、予測モデルのラベル１予測精度は、Recall：0.06、Precision：0.82 程度です（2025年3月現在）。
つまり、ラベル１データの発見率は6％、発見したデータが真にラベル１である確率は82％です。

通常であれば、このように「見逃し」の多い（6％しか発見できない！）偏ったモデルは実用に適しません。
しかし、「株取引で儲ける」という目的であれば、十分に実用可能です。
すくい上げたチャンスがわずかであっても、そのうちの82％が「本当のチャンス」なので、これを確実に掴めばよいからです。

このように、本プログラムは、不確かな予測でも儲かるアルゴリズムを追求しています。
具体的には、

1. 「値上がりしそうな銘柄」を絞り込む
2. 絞り込んだ銘柄について始値で買って終値で売る
3. いくら儲かったかを検証する

というオペレーションを実行します。

## 1. 「値上がりしそうな銘柄」を絞り込む

東証に上場している約 4000 銘柄から、

- ETF や REIT などの非企業の銘柄
- 上場から間がなくデータが不十分な銘柄
- 出来高が少なく取引に向かない銘柄

を除外すると、約 2000 銘柄になります。このなかから「当日の終値を基準にして、翌営業日の終値が 0.5 ％超上がる銘柄」を、予測モデルで発見します。
具体的には、次の4ステップで 2000 銘柄を 50 銘柄に絞り込みます。

### 1-1. 直近30日分の株価から入力データを構成する

Yahoo! ファイナンスから取得した株価データを利用して、1日あたり 21 次元のデータを構成します。これを30日分束ねた 21x30 の行列がモデルの入力となります。

### 1-2. 翌営業日の株価から正解データを構成する

翌営業日の終値が当日の終値から 0.5 ％上昇していたら、クラス1のラベルを付与します。

### 1-3. 予測モデルを学習させる

21x30 の行列とそれに対応するラベルのペアを与えて、予測モデルを学習させます。なお、モデルは6層からなる DNN で、2層目に双方向 RNN を利用しています。

```python
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

### 1-4. 上記 1.1〜1.3 を直近 31〜34 日分の株価を用いて同様に行う

まず、直近 31 日分の株価から入力データを構成し、正解データを与え、別のモデルを学習させます。同様に、32〜34 日分の株価を用いてそれぞれのモデルを学習させます。したがって、予測モデルは全部で５つ完成します。ここまでの処理を担っているのが、```update_model.py``` です。

### 1-5. 当日までのデータと５つの予測モデルを使って翌営業日の値上がりを予測する

モデルが出力する予測値（0〜1 の値をとる）に基づいて、各銘柄をラベル 0 または 1 に振り分けます。具体的には、0.7 を下限の閾値としてこれを徐々に下げながら、2000 銘柄を最大 50 銘柄に絞り込みます。
0.7 を下限とするため、場合によっては選抜された銘柄の数が 50 に達さないことがありますし、0.7 より高い閾値のまま 50 銘柄に達することもあります。

私の経験上、50 に達さない場合（例えば、10 銘柄ほどしか残らない場合など）は、翌日は市場全体が下げ相場になることが多く、閾値が高いまま 50 に達する場合は、上げ相場になることが多いようです。
この「絞り込み」の処理を担っているのが、```predictor.py``` です。

なお、絞り込む数を最大 50 とするのは、kabu ステーション API の利用制限（取引対象となる「登録銘柄」は 50 銘柄以下）があるためです。また、予測モデルを５つ使うのは、低い Recall をカバーして取りこぼしを少なくしたいという意図によるものです（本当に少なくなっているか＝Recall が改善されているかは未検証）。

## 2. 絞り込んだ銘柄について始値で買って終値で売る

予測モデルで絞り込まれた銘柄を対象に、始値で買って（寄付成行買い建て）終値で売ります（引け成行返済）。この売買に関する処理を担っているのが、```stocktrading.py``` と ```stock.py``` です。

予測モデルは「前日の終値→当日の終値」で学習・予測しているのに、実際は「当日の始値→当日の終値」で売買しています。モデルの学習・予測に「前日の終値（実際は買うことができない値段）」を使っている以上、この矛盾は解消できません。これは今後の課題です。

## 3. いくら儲かったかを検証する

各銘柄について取引を実行した当日の始値・終値の情報を、[kabutan.jp](https://kabutan.jp/) をクローリングして入手し、いくら儲かったかを検証します。この処理を担うのが、```crawler.py``` です。

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
上記のように設定ファイル（nginx.conf）を書き換えたら、nginxを起動します。

## WindowsマシンのローカルIPアドレスを調べる

WindowsマシンのローカルIPアドレスを調べてください。コマンドプロンプトで```$ ipconfig /all```と打てば表示されます。

## .envファイルを書き換える

本プログラムのディレクトリ直下にある```.env```を下記のように書き換えてください。

```:.env
APIPassword_production=XXXXX ← 証券会社から付与されたAPIパスワード
IPAddress=192.168.0.ZZZ ← 先ほど調べたWindowsマシンのローカルIPアドレス
Port=80
BaseDir=/path/to/dir/stocktrading
```

## プログラムを実行する

とりあえず、```$ make``` してみましょう。うまくいけば、取引余力が表示されるはずです。

## その他

kabuステーションは毎日再起動する必要があることに注意しましょう。\
APIキーは毎日リフレッシュ（更新）する必要があるのですが、kabuステーションを再起動しなければ、これができないようになっています。

なぜこのクソみたいな仕様になっているのかは分かりません。




