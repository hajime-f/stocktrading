import datetime
import re
from logging import Formatter, LogRecord

import jpholiday
import yaml


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

    def count_business_days(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> int:
        """
        指定された期間内の日本の営業日を数える。
        （入力はdate型、休日判定はMisc.check_day_typeを前提とする）

        Args:
        start_date (datetime.date): 開始日。
        end_date (datetime.date): 終了日。

        Returns:
        int: 期間内の営業日の日数。開始日が終了日より後の場合は0を返す。
        """
        # 1. 開始日と終了日の順序をチェック
        if start_date > end_date:
            return 0

        # 2. 営業日をカウント
        business_days_count = 0
        current_date = start_date
        one_day = datetime.timedelta(days=1)
        misc_checker = Misc()  # 判定用のインスタンスを作成

        while current_date <= end_date:
            # 与えられた関数を使って、平日(0)かどうかを判定
            if misc_checker.check_day_type(current_date) == 0:
                business_days_count += 1

            # 次の日に進める
            current_date += one_day

        return business_days_count

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


class MessageManager:
    def __init__(self, file_path="log_messages.yaml"):
        try:
            with open(file_path, "rt", encoding="utf-8") as f:
                self._messages = yaml.safe_load(f)
        except FileNotFoundError:
            self._messages = {}
            print(f"警告: メッセージ定義ファイル '{file_path}' が見つかりません。")

    def get(self, key: str, **kwargs) -> str:
        try:
            template = self._messages
            for k in key.split("."):
                template = template[k]

            if not isinstance(template, str):
                return f"メッセージキー '{key}' は有効な文字列ではありません。"

            return template.format(**kwargs)

        except (KeyError, TypeError):
            return f"メッセージキー '{key}' が見つかりませんでした。"
