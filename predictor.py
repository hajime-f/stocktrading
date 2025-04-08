import datetime
import pandas as pd
import sqlite3

from data_manager import DataManager
from misc import Misc
from models import LongModel, ShortModel, ThresholdModel

if __name__ == "__main__":
    # 土日祝日は実行しない
    misc = Misc()
    if misc.check_day_type(datetime.date.today()):
        exit()

    threshold_model = ThresholdModel()
    long_model = LongModel()
    short_model = ShortModel()

    # 学習
    percentage = 0.5
    df_threshold_model_names = threshold_model.fit(percentage)
    df_long_model_names = long_model.fit(percentage)
    df_short_model_names = short_model.fit(percentage)

    # 予測
    df_threshold = threshold_model.predict(df_threshold_model_names)
    df_long = long_model.predict(df_long_model_names)
    df_short = short_model.predict(df_short_model_names)

    # nbd = datetime.date.today()
    nbd = misc.get_next_business_day(datetime.date.today())
    df = pd.DataFrame(
        {
            "date": nbd.strftime("%Y-%m-%d"),
            "threshold": df_threshold["pred"].mean(),
            "long": df_long["pred"].mean(),
            "short": df_short["pred"].mean(),
            "pl_long": None,
            "pl_short": None,
        },
        index=[0],
    )
    dm = DataManager()
    conn = sqlite3.connect(dm.db)
    with conn:
        df.to_sql("Aggregate", conn, if_exists="append", index=False)
