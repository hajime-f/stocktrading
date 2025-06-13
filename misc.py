import datetime
import re
from logging import Formatter, LogRecord

import jpholiday


class Misc:
    def __init__(self):
        pass

    def check_day_type(self, date_input):
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

    def get_next_business_day(self, date_input):
        """
        指定された日付の次の営業日（土日祝日を除く平日）を返す。

        Args:
        date_input (datetime.date | datetime.datetime | str):
        基準となる日付。datetime.dateオブジェクト、datetime.datetimeオブジェクト、
        または "YYYY-MM-DD" 形式の文字列。

        Returns:
        datetime.date: 次の営業日の日付オブジェクト。
                    無効な入力の場合は ValueError を発生させます。
        """
        try:
            if isinstance(date_input, datetime.datetime):
                base_date = date_input.date()
            elif isinstance(date_input, datetime.date):
                base_date = date_input
            elif isinstance(date_input, str):
                base_date = datetime.date.fromisoformat(date_input)
            else:
                raise ValueError("無効な日付入力タイプです。")
        except ValueError as e:
            raise ValueError(f"無効な日付形式です: {e}") from e

        # 基準日の翌日からチェックを開始
        next_day = base_date + datetime.timedelta(days=1)

        while True:
            # 曜日チェック (月=0, ..., 土=5, 日=6)
            is_weekday = next_day.weekday() < 5
            # 祝日チェック
            is_not_holiday = not jpholiday.is_holiday(next_day)

            # 平日であればその日付を返す
            if is_weekday and is_not_holiday:
                return next_day
            else:
                # 平日でなければ次の日へ
                next_day += datetime.timedelta(days=1)


class StripRichFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        formatted_log = super().format(record)
        stripped_log = re.sub(r"\[(/?[\w\s#]*)\]", "", formatted_log)

        return stripped_log
