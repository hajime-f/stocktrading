import datetime

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

    breakpoint()
