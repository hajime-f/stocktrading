import datetime
import jpholiday


class Misc:
    def __init__(self):
        pass


def check_day_type(date_input):
    """
    与えられた日付が平日、土日、祝日のいずれかを判定する。

    Args:
        date_input (datetime.date | datetime.datetime | str):
            判定したい日付。datetime.dateオブジェクト、datetime.datetimeオブジェクト、
            または "YYYY-MM-DD" 形式の文字列。

    Returns:
        str: "平日"、"土日"、"祝日" のいずれか。
             日付の形式が無効な場合は "無効な日付" を返す。
    """
    try:
        if isinstance(date_input, datetime.datetime):
            date_obj = date_input.date()
        elif isinstance(date_input, datetime.date):
            date_obj = date_input
        elif isinstance(date_input, str):
            date_obj = datetime.date.fromisoformat(date_input)
        else:
            return None
    except ValueError:
        return None

    # 祝日判定
    if jpholiday.is_holiday(date_obj):
        return 1
    # 土日判定
    elif date_obj.weekday() >= 5:
        return 1
    # 上記以外は平日
    else:
        return 0
