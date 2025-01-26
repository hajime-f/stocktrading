# kabu station API を使った日本株の自動売買プログラム

kabu ステーション API を使って、日本株を自動的に売買するプログラムです。

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
Port=:18080
```

nginx のリバースプロキシ機能を使って、ローカルマシン（kabu ステーションが動作しているマシン）と、リモートマシン（本プログラムを動作させるマシン）とを分けている場合は、ローカルマシンの IP アドレスを指定してください。


### パッケージのインストール

