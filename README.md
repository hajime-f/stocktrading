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
Port=:18080
```

### パッケージのインストールと実行

仮想環境に入って ```make install``` した後、```make``` すればプログラムが動きます。

```
$ source env/bin/activate
(env) $ make install
(env) $ make
```



