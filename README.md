# kabu station API を使った日本株の自動売買プログラム

[kabuステーションAPI](https://kabucom.github.io/kabusapi/ptal/) を使って、日本株を自動的に売買するプログラムです。

> [!CAUTION]
> 本プログラムを使用することによって被った損害等について、制作者は一切の責任を負いません。
> 投資はご自身の判断と責任のもとで行ってください。

# 使い方

## .env ファイルの追加

プログラムディレクトリの直下に、下記のようにパスワード等を記録した .env ファイルを配置してください。

```:.env
APIPassword_production=XXXX
OrderPassword=YYYY
IPAddress=127.0.0.1
Port=18080
```

## 仮想環境の作成

仮想環境を作ってください。

```
$ python -m venv env
```

## パッケージのインストール

仮想環境に入って ```make install``` してください。

```
$ source ./env/bin/activate
(env) $ make install
```
## データの収集

下記のコマンドを打って、予測モデルを学習させるためのデータを集めてください。

```
(env) $ make collect
```

15:30 になると PUSH 配信が止まるため、端末の表示も止まります。\
ほとぼりが冷めたくらいを見計らって、```Ctrl+C``` でプログラムを停止してください。\
ただし、```Ctrl+C``` を押してからプロンプトが返ってくるまで時間がかかります。辛抱強く待ってください。

データは ```./data``` 配下に生成されます。少なくとも３日分くらいのデータを集めましょう。

## 予測モデルの学習

予測モデルを学習させます。\
まず、```train_model.py``` の８行目に、集めたデータのファイル名を列挙してください。

```python
filename_list = ['data_20250206_201528.pkl', 'data_20250207_201857.pkl', ... , 'data_20250212_192845.pkl']
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

# API利用までの流れとMacを使った開発環境の構築方法

「kabuステーションAPI」は、[「kabuステーション」](https://kabu.com/kabustation/default.html)と呼ばれる株取引専用ソフトウェアを経由しなければ使用できません。つまり、原則としてkabuステーションが動作するPC上でプログラムを実行することが求められ、このとき「localhost」をホストに指定して各エンドポイントにアクセスすることになります。

しかし、残念ながら、kabuステーションはWindows版しか提供されていません。\
そのため、Macでプログラムを開発・実行するためには、テクニカルな工夫が必要になります。

下記では、証券口座の開設からAPIの利用設定、そしてMacにおける開発環境の構築方法について説明します。

### 三菱UFJ eスマート証券に証券口座を開設する

[トップページ](https://kabu.com/)から「口座開設」を選択し、ガイダンスに沿って必要事項を入力していけば口座が開設できます。\
なお、口座は「特定口座 源泉徴収あり」にしておきましょう。「源泉徴収なし」にすると確定申告が面倒なので。

### kabuステーションをインストールする

Windowsマシンにkabuステーションをインストールします。\
なお、Windowsマシンは実機である必要はありません。私は[Parallels](https://www.parallels.com/jp/)上で動作する仮想Windowsマシンでkabuステーションを動作させています。

### APIの利用設定を行う

[このガイダンス](https://kabucom.github.io/kabusapi/ptal/howto.html)に沿って利用設定を進めます。\
しかし、kabuステーションAPIの「状態」が「利用可」にならず、途中でつまづくと思います。

これは、APIを利用するためには「Professionalプラン」である必要があるのですが、現段階ではその条件を満たしておらず、「通常プラン」でしかないからです。

### Professionalプラン移行のための条件を満たす

[このページ](https://kabu.com/tool/kabustation/default.html)の「ご利用プランについて」の記載によれば、Professionalプラン移行には次の2つの条件を満たす必要があります。

1. 信用取引口座または、先物オプション取引口座開設済み
2. 前々々月～前営業日で当社全取引における約定回数が1回以上ある

1については信用取引の口座を開設するだけなので簡単なのですが、問題は2です。とにかくなんでもいいので株取引を少なくとも１回約定させ、取引実績を作る必要があります。

私は、銘柄1475（iシェアーズ・コア TOPIX ETF）を買ってすぐ売却しました。この銘柄は値段が安く、10株単位で取引できるので、数千円あれば取引実績を作ることができます。

### APIの利用設定を行う（再）

約定の翌営業日に、APIの利用設定ができるようになっているはずです。

### APIシステム設定を行う

kabuステーションを起動し、[このページ](https://kabucom.github.io/kabusapi/ptal/howto.html)のガイダンスに沿って「APIシステム設定」を進めます。このガイダンスに記載のとおり、kabuステーションの画面右上のアイコンが緑色になっていることを確認しましょう。

### nginx をインストールする

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

### WindowsマシンのローカルIPアドレスを調べる

WindowsマシンのローカルIPアドレスを調べてください。コマンドプロンプトで```$ ipconfig /all```と打てば表示されます。

### .envファイルを書き換える

本プログラムのディレクトリ直下にある```.env```を下記のように書き換えてください。

```
APIPassword_production=XXXXX ← 証券会社から付与されたAPIパスワード
OrderPassword=YYYYY ← 証券会社から付与された取引パスワード
IPAddress=192.168.0.ZZZ ← 先ほど調べたWindowsマシンのローカルIPアドレス
Port=80
```

### プログラムを実行する

とりあえず、```$ make``` してみましょう。うまくいけば、買付余力が表示されるはずです。

### その他

kabuステーションは毎日再起動する必要があることに注意しましょう。\
APIキーは毎日リフレッシュ（更新）する必要があるのですが、kabuステーションを再起動しなければ、これができないようになっています。

なぜこのクソみたいな仕様になっているのかは分かりません。



