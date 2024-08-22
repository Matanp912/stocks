import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
import pywhatkit as pwt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from imblearn.over_sampling import ADASYN, SMOTE


# PARAMETERS:
WHATSAPP_ID = "******************"
PHONE_NUMBER = "+***************"

columns_to_merge_after_normalized = [
    "did_break_highest_all_time_amzn",
    "did_break_highest_all_time_tsla",
    "did_break_highest_all_time_msft",
    "did_break_highest_all_time_fb",
    "did_break_highest_all_time_googl",
    "did_break_highest_all_time_aapl",
    "did_break_highest_all_time_tecs",
    "did_break_highest_all_time_spy",
    "did_break_highest_all_time_bac",
    "did_break_highest_all_time_ndaq",
    "Positive_change_in_the_last_3_day",
    "Positive_change_in_the_last_7_day",
    "did_break_highest_all_time_tqqq",
    "After Covid",
    "Weekday",
]
columns_to_normalize = [
    "change_in_percentage_for_7_day_amzn",
    "Average_volume_for_7_day_amzn",
    "Average_close_price_for_7_day_amzn",
    "days_in_a_row_for_trend_amzn",
    "previous_days_in_a_row_for_trend_amzn",
    "change_in_percentage_for_7_day_tsla",
    "Average_volume_for_7_day_tsla",
    "Average_close_price_for_7_day_tsla",
    "days_in_a_row_for_trend_tsla",
    "previous_days_in_a_row_for_trend_tsla",
    "change_in_percentage_for_7_day_msft",
    "Average_volume_for_7_day_msft",
    "Average_close_price_for_7_day_msft",
    "days_in_a_row_for_trend_msft",
    "previous_days_in_a_row_for_trend_msft",
    "change_in_percentage_for_7_day_fb",
    "Average_volume_for_7_day_fb",
    "Average_close_price_for_7_day_fb",
    "days_in_a_row_for_trend_fb",
    "previous_days_in_a_row_for_trend_fb",
    "change_in_percentage_for_7_day_googl",
    "Average_volume_for_7_day_googl",
    "Average_close_price_for_7_day_googl",
    "days_in_a_row_for_trend_googl",
    "previous_days_in_a_row_for_trend_googl",
    "change_in_percentage_for_7_day_aapl",
    "Average_volume_for_7_day_aapl",
    "Average_close_price_for_7_day_aapl",
    "days_in_a_row_for_trend_aapl",
    "previous_days_in_a_row_for_trend_aapl",
    "change_in_percentage_for_7_day_spy",
    "Average_close_price_for_7_day_spy",
    "days_in_a_row_for_trend_spy",
    "previous_days_in_a_row_for_trend_spy",
    "change_in_percentage_for_7_day_uvxy",
    "Average_close_price_for_7_day_uvxy",
    "days_in_a_row_for_trend_uvxy",
    "previous_days_in_a_row_for_trend_uvxy",
    "change_in_percentage_for_7_day_tqqq",
    "Average_close_price_for_7_day_tqqq",
    "days_in_a_row_for_trend_tqqq",
    "previous_days_in_a_row_for_trend_tqqq",
    "change_in_percentage_for_7_day_bac",
    "Average_volume_for_7_day_bac",
    "Average_close_price_for_7_day_bac",
    "days_in_a_row_for_trend_bac",
    "previous_days_in_a_row_for_trend_bac",
    "change_in_percentage_for_7_day_ndaq",
    "Average_volume_for_7_day_ndaq",
    "Average_close_price_for_7_day_ndaq",
    "days_in_a_row_for_trend_ndaq",
    "previous_days_in_a_row_for_trend_ndaq",
    "change_in_percentage_for_7_day_tecs",
    "Average_volume_for_7_day_tecs",
    "Average_close_price_for_7_day_tecs",
    "days_in_a_row_for_trend_tecs",
    "previous_days_in_a_row_for_trend_tecs",
]
columns_for_x = [
    "did_break_highest_all_time_amzn",
    "did_break_highest_all_time_tsla",
    "did_break_highest_all_time_msft",
    "did_break_highest_all_time_fb",
    "did_break_highest_all_time_googl",
    "did_break_highest_all_time_aapl",
    "did_break_highest_all_time_bac",
    "did_break_highest_all_time_ndaq",
    "did_break_highest_all_time_tecs",
    "Positive_change_in_the_last_3_day",
    "Positive_change_in_the_last_7_day",
    "did_break_highest_all_time_tqqq",
    "did_break_highest_all_time_spy",
    "change_in_percentage_for_7_day_amzn",
    "Average_volume_for_7_day_amzn",
    "Average_close_price_for_7_day_amzn",
    "days_in_a_row_for_trend_amzn",
    "previous_days_in_a_row_for_trend_amzn",
    "change_in_percentage_for_7_day_tsla",
    "Average_volume_for_7_day_tsla",
    "Average_close_price_for_7_day_tsla",
    "days_in_a_row_for_trend_tsla",
    "previous_days_in_a_row_for_trend_tsla",
    "change_in_percentage_for_7_day_msft",
    "Average_volume_for_7_day_msft",
    "Average_close_price_for_7_day_msft",
    "days_in_a_row_for_trend_msft",
    "previous_days_in_a_row_for_trend_msft",
    "change_in_percentage_for_7_day_fb",
    "Average_volume_for_7_day_fb",
    "Average_close_price_for_7_day_fb",
    "days_in_a_row_for_trend_fb",
    "previous_days_in_a_row_for_trend_fb",
    "change_in_percentage_for_7_day_googl",
    "Average_volume_for_7_day_googl",
    "Average_close_price_for_7_day_googl",
    "days_in_a_row_for_trend_googl",
    "previous_days_in_a_row_for_trend_googl",
    "change_in_percentage_for_7_day_aapl",
    "Average_volume_for_7_day_aapl",
    "Average_close_price_for_7_day_aapl",
    "days_in_a_row_for_trend_aapl",
    "previous_days_in_a_row_for_trend_aapl",
    "change_in_percentage_for_7_day_spy",
    "Average_close_price_for_7_day_spy",
    "days_in_a_row_for_trend_spy",
    "previous_days_in_a_row_for_trend_spy",
    "change_in_percentage_for_7_day_uvxy",
    "Average_close_price_for_7_day_uvxy",
    "days_in_a_row_for_trend_uvxy",
    "previous_days_in_a_row_for_trend_uvxy",
    "change_in_percentage_for_7_day_tqqq",
    "Average_close_price_for_7_day_tqqq",
    "days_in_a_row_for_trend_tqqq",
    "previous_days_in_a_row_for_trend_tqqq",
    "change_in_percentage_for_7_day_bac",
    "Average_volume_for_7_day_bac",
    "Average_close_price_for_7_day_bac",
    "days_in_a_row_for_trend_bac",
    "previous_days_in_a_row_for_trend_bac",
    "change_in_percentage_for_7_day_ndaq",
    "Average_volume_for_7_day_ndaq",
    "Average_close_price_for_7_day_ndaq",
    "days_in_a_row_for_trend_ndaq",
    "previous_days_in_a_row_for_trend_ndaq",
    "change_in_percentage_for_7_day_tecs",
    "Average_volume_for_7_day_tecs",
    "Average_close_price_for_7_day_tecs",
    "days_in_a_row_for_trend_tecs",
    "previous_days_in_a_row_for_trend_tecs",
    "After Covid",
    "Weekday",
]
column_for_y = ["label_1_day"]
positive_column_for_real_money_check = "tqqq_change_for_1_day"
negative_column_for_real_money_check = "sqqq_change_for_1_day"
new_columns_for_x_temp = columns_for_x.copy()
remove_lst = [
    "did_break_highest_all_time_amzn",
    "did_break_highest_all_time_tsla",
    "did_break_highest_all_time_msft",
    "did_break_highest_all_time_fb",
    "did_break_highest_all_time_googl",
    "did_break_highest_all_time_aapl",
    "did_break_highest_all_time_bac",
    "did_break_highest_all_time_ndaq",
    "did_break_highest_all_time_tecs",
    "Positive_change_in_the_last_7_day",
    "Positive_change_in_the_last_3_day",
    "did_break_highest_all_time_tqqq",
    "did_break_highest_all_time_spy",
    "change_in_percentage_for_7_day_amzn",
    "Average_volume_for_7_day_amzn",
    "Average_close_price_for_7_day_amzn",
    "days_in_a_row_for_trend_amzn",
    "previous_days_in_a_row_for_trend_amzn",
    "change_in_percentage_for_7_day_tsla",
    "Average_volume_for_7_day_tsla",
    "Average_close_price_for_7_day_tsla",
    "days_in_a_row_for_trend_tsla",
    "previous_days_in_a_row_for_trend_tsla",
    "Average_volume_for_7_day_msft",
    "Average_close_price_for_7_day_msft",
    "previous_days_in_a_row_for_trend_msft",
    "change_in_percentage_for_7_day_fb",
    "Average_volume_for_7_day_fb",
    "Average_close_price_for_7_day_fb",
    "previous_days_in_a_row_for_trend_fb",
    "change_in_percentage_for_7_day_googl",
    "Average_volume_for_7_day_googl",
    "Average_close_price_for_7_day_googl",
    "days_in_a_row_for_trend_googl",
    "previous_days_in_a_row_for_trend_googl",
    "change_in_percentage_for_7_day_aapl",
    "Average_volume_for_7_day_aapl",
    "Average_close_price_for_7_day_aapl",
    "days_in_a_row_for_trend_aapl",
    "previous_days_in_a_row_for_trend_aapl",
    "change_in_percentage_for_7_day_spy",
    "Average_close_price_for_7_day_spy",
    "days_in_a_row_for_trend_spy",
    "previous_days_in_a_row_for_trend_spy",
    "change_in_percentage_for_7_day_uvxy",
    "Average_close_price_for_7_day_uvxy",
    "days_in_a_row_for_trend_uvxy",
    "previous_days_in_a_row_for_trend_uvxy",
    "Average_close_price_for_7_day_tqqq",
    "change_in_percentage_for_7_day_bac",
    "Average_volume_for_7_day_bac",
    "Average_close_price_for_7_day_bac",
    "days_in_a_row_for_trend_bac",
    "previous_days_in_a_row_for_trend_bac",
    "change_in_percentage_for_7_day_ndaq",
    "Average_volume_for_7_day_ndaq",
    "Average_close_price_for_7_day_ndaq",
    "days_in_a_row_for_trend_ndaq",
    "Average_volume_for_7_day_tecs",
    "Average_close_price_for_7_day_tecs",
    "days_in_a_row_for_trend_tecs",
    "previous_days_in_a_row_for_trend_tecs",
    "After Covid",
]
for i in remove_lst:
    new_columns_for_x_temp.remove(i)
XGB_clf = XGBClassifier(verbosity=0)

group_name = "test group"
private_group_name = "Model documendation"
no_trading_dates = [
    "2021-09-06",
    "2021-11-25",
    "2021-12-24",
    "2022-01-17",
    "2022-02-21",
    "2022-04-15",
    "2022-05-30",
    "2022-06-20",
    "2022-07-04",
    "2022-09-05",
    "2022-11-24",
    "2022-12-26",
]


def main_before_23(
    final_train_data_for_prediction,
    final_train_lables_for_prediction,
    final_standard_scaler,
    public=True,
):
    if public:
        pwt.manual_send_message_to_someone(group_name, "Making the first decision...")

    before_midnight = True
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        merged_dfs,
    ) = organized_data_for_prediction(before_midnight, yf_df=True)
    x_test, _, _ = predict_today(
        merged_dfs,
        columns_for_x,
        positive_column_for_real_money_check,
        negative_column_for_real_money_check,
        final_standard_scaler,
        columns_to_normalize,
        columns_to_merge_after_normalized,
        new_columns_for_x_temp,
        pca=False,
    )

    todays_row = pd.DataFrame(x_test.iloc[-1]).T
    no_trading_dates = [
        "2021-09-06",
        "2021-11-25",
        "2021-12-24",
        "2022-01-17",
        "2022-02-21",
        "2022-04-15",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2022-12-26",
    ]

    while True:
        if todays_row.index in no_trading_dates:
            todays_row.index = todays_row.index + timedelta(days=1)
            todays_row["Weekday"] += 1
            if list(todays_row["Weekday"])[0] == 5:
                todays_row.index = todays_row.index + timedelta(days=2)
                todays_row["Weekday"] = 0
            elif list(todays_row["Weekday"])[0] == 6:
                todays_row.index = todays_row.index + timedelta(days=1)
                todays_row["Weekday"] = 0
        else:
            break
    print("todays_row")
    print(todays_row)
    proba, action = predict_todays_direction(
        XGB_clf,
        final_train_data_for_prediction,
        todays_row,
        final_train_lables_for_prediction,
        threshold=0.15,
    )

    return action, proba, todays_row


def create_yf_df():
    today = str(datetime.today() + timedelta(days=1)).split(" ")[0]
    data = yf.download(
        "AMZN TSLA MSFT FB GOOGL AAPL SPY TQQQ SQQQ UVXY BAC NDAQ TECS",
        start="2012-06-01",
        end=today,
    )
    if (data.iloc[-1, 25] / data.iloc[-2, 25]) < 0.7:
        data.iloc[-1:, 25] *= 2
        data.iloc[-1:, 23] /= 5

    close_data = data["Close"].rename(
        columns={
            "AMZN": "AMZN_close",
            "TSLA": "TSLA_close",
            "MSFT": "MSFT_close",
            "FB": "FB_close",
            "GOOGL": "GOOGL_close",
            "AAPL": "AAPL_close",
            "SPY": "SPY_close",
            "UVXY": "UVXY_close",
            "TQQQ": "TQQQ_close",
            "SQQQ": "SQQQ_close",
            "BAC": "BAC_close",
            "NDAQ": "NDAQ_close",
            "TECS": "TECS_close",
        }
    )
    Volume_data = data["Volume"].rename(
        columns={
            "AMZN": "AMZN_volume",
            "TSLA": "TSLA_volume",
            "MSFT": "MSFT_volume",
            "FB": "FB_volume",
            "GOOGL": "GOOGL_volume",
            "AAPL": "AAPL_volume",
            "SPY": "SPY_volume",
            "UVXY": "UVXY_volume",
            "TQQQ": "TQQQ_volume",
            "SQQQ": "SQQQ_volume",
            "BAC": "BAC_volume",
            "NDAQ": "NDAQ_volume",
            "TECS": "TECS_volume",
        }
    )

    yf_df = pd.concat([close_data, Volume_data], axis=1).reset_index()
    yf_df["AMZN"] = ""
    yf_df["TSLA"] = ""
    yf_df["MSFT"] = ""
    yf_df["FB"] = ""
    yf_df["GOOGL"] = ""
    yf_df["AAPL"] = ""
    yf_df["SPY"] = ""
    yf_df["TQQQ"] = ""
    yf_df["SQQQ"] = ""
    yf_df["UVXY"] = ""
    yf_df["BAC"] = ""
    yf_df["NDAQ"] = ""
    yf_df["TECS"] = ""

    yf_df = yf_df[
        [
            "AMZN",
            "Date",
            "AMZN_close",
            "Date",
            "AMZN_volume",
            "TSLA",
            "Date",
            "TSLA_close",
            "Date",
            "TSLA_volume",
            "MSFT",
            "Date",
            "MSFT_close",
            "Date",
            "MSFT_volume",
            "FB",
            "Date",
            "FB_close",
            "Date",
            "FB_volume",
            "GOOGL",
            "Date",
            "GOOGL_close",
            "Date",
            "GOOGL_volume",
            "AAPL",
            "Date",
            "AAPL_close",
            "Date",
            "AAPL_volume",
            "SPY",
            "Date",
            "SPY_close",
            "TQQQ",
            "Date",
            "TQQQ_close",
            "SQQQ",
            "Date",
            "SQQQ_close",
            "UVXY",
            "Date",
            "UVXY_close",
            "BAC",
            "Date",
            "BAC_close",
            "Date",
            "BAC_volume",
            "NDAQ",
            "Date",
            "NDAQ_close",
            "Date",
            "NDAQ_volume",
            "TECS",
            "Date",
            "TECS_close",
            "Date",
            "TECS_volume",
        ]
    ].rename(
        columns={
            "AMZN_close": "Close",
            "AMZN_volume": "Volume",
            "TSLA_close": "Close",
            "TSLA_volume": "Volume",
            "MSFT_close": "Close",
            "MSFT_volume": "Volume",
            "FB_close": "Close",
            "FB_volume": "Volume",
            "GOOGL_close": "Close",
            "GOOGL_volume": "Volume",
            "AAPL_close": "Close",
            "AAPL_volume": "Volume",
            "SPY_close": "Close",
            "UVXY_close": "Close",
            "TQQQ_close": "Close",
            "SQQQ_close": "Close",
            "BAC_close": "Close",
            "BAC_volume": "Volume",
            "NDAQ_close": "Close",
            "NDAQ_volume": "Volume",
            "TECS_close": "Close",
            "TECS_volume": "Volume",
        }
    )
    return yf_df


def change_date(date):
    day, month, year = date[0:10].split("/")
    return pd.to_datetime(year + month + day)


def change_date_for_df(dfs):
    for df in dfs:
        df["Date"] = df["Date"].apply(change_date)
        if df.shape[1] == 3:
            df["Volume"] = df["Volume"].astype(float)
        df["Close"] = df["Close"].astype(float)


def calculate_change_in_percentage(df, days, row):
    i = row.name
    if i - days > 0:
        current_price = df.iloc[i - 1, :]["Close"]
        past_price = df.iloc[i - 1 - days]["Close"]

        return (current_price / past_price - 1) * 100
    else:
        return np.nan


def calculate_change_in_percentage_for_dfs(dfs, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["change_in_percentage_for_1_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_change_in_percentage(temp, 1, row), axis=1
        )
        temp["change_in_percentage_for_7_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_change_in_percentage(temp, 7, row), axis=1
        )
        temp["change_in_percentage_for_30_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_change_in_percentage(temp, 30, row), axis=1
        )


def calculate_average_volume(df, days, row):

    i = row.name
    if i - days > 0:
        mean_volume = df.iloc[i - days : i, :]["Volume"].mean()

        return mean_volume
    else:
        return np.nan


def calculate_average_volume_for_dfs(dfs, dfs_names):
    for i in range(len(dfs)):
        if dfs_names[i] in ["tqqq", "qqq", "spy", "uvxy", "sqqq"]:
            continue
        temp = dfs[i]
        temp["Average_volume_for_1_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_volume(temp, 1, row), axis=1
        )
        temp["Average_volume_for_7_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_volume(temp, 7, row), axis=1
        )
        temp["Average_volume_for_30_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_volume(temp, 30, row), axis=1
        )


def calculate_average_close_price(df, days, row):

    i = row.name
    if i - days > 0:
        mean_price = df.iloc[i - days : i, :]["Close"].mean()

        return mean_price
    else:
        return np.nan


def calculate_average_close_price_for_dfs(dfs, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["Average_close_price_for_1_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_close_price(temp, 1, row), axis=1
        )
        temp["Average_close_price_for_7_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_close_price(temp, 7, row), axis=1
        )
        temp["Average_close_price_for_30_day_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_average_close_price(temp, 30, row), axis=1
        )


def add_label_x_days_before(df, days, row):
    i = row.name
    if i - days > 0:
        current_price = df.iloc[i - 1, :]["Close"]
        past_price = df.iloc[i - 1 - days]["Close"]

        return int(current_price > past_price)
    else:
        return np.nan


def add_label_x_days_before_to_qqq_df(qqq_df, sqqq=False):
    if not sqqq:
        qqq_df["Positive_change_in_the_last_1_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 1, row), axis=1
        )
        qqq_df["Positive_change_in_the_last_3_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 3, row), axis=1
        )

        qqq_df["Positive_change_in_the_last_7_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 7, row), axis=1
        )
        qqq_df["Positive_change_in_the_last_30_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 30, row), axis=1
        )
    else:
        qqq_df["sqqq_Positive_change_in_the_last_1_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 1, row), axis=1
        )
        qqq_df["sqqq_Positive_change_in_the_last_3_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 3, row), axis=1
        )

        qqq_df["sqqq_Positive_change_in_the_last_7_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 7, row), axis=1
        )
        qqq_df["sqqq_Positive_change_in_the_last_30_day"] = qqq_df.apply(
            lambda row: add_label_x_days_before(qqq_df, 30, row), axis=1
        )


def add_label_x_days_to_the_future(df, days, row):
    i = row.name
    if i - 1 < 0:
        return np.nan
    if i + days <= df.shape[0]:
        current_price = df.iloc[i - 1, :]["Close"]
        future_price = df.iloc[i - 1 + days]["Close"]
        return int(future_price > current_price)
    else:
        return np.nan


def add_label_x_days_to_the_future_to_qqq_df(qqq_df):
    qqq_df["label_1_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 1, row), axis=1
    )
    qqq_df["label_7_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 7, row), axis=1
    )
    qqq_df["label_30_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 30, row), axis=1
    )


def calculate_change_for_x_days_to_the_future(df, days, row):
    i = row.name
    if i - 1 < 0:
        return np.nan
    if i + days <= df.shape[0]:
        current_price = df.iloc[i - 1, :]["Close"]
        future_price = df.iloc[i - 1 + days]["Close"]
        return (future_price / current_price - 1) * 100
    else:
        return np.nan


def add_change_for_x_days_to_the_future(qqq_df, sqqq=False):
    if not sqqq:
        qqq_df["tqqq_change_for_1_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 1, row),
            axis=1,
        )
        qqq_df["tqqq_change_for_7_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 7, row),
            axis=1,
        )
        qqq_df["tqqq_change_for_30_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 30, row),
            axis=1,
        )
    else:
        qqq_df["sqqq_change_for_1_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 1, row),
            axis=1,
        )
        qqq_df["sqqq_change_for_7_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 7, row),
            axis=1,
        )
        qqq_df["sqqq_change_for_30_day"] = qqq_df.apply(
            lambda row: calculate_change_for_x_days_to_the_future(qqq_df, 30, row),
            axis=1,
        )


def calculate_trend(df, row):
    trend = 0
    counter = 0
    i = row.name
    if i < 2:
        return np.nan
    for j in range(i - 1, -1, -1):
        if counter > 0:
            if df.iloc[j]["Close"] >= df.iloc[j - 1]["Close"] and trend > 0:
                trend += 1
            elif df.iloc[j]["Close"] <= df.iloc[j - 1]["Close"] and trend < 0:
                trend -= 1
            else:
                break

        else:
            if df.iloc[j]["Close"] >= df.iloc[j - 1]["Close"]:
                trend += 1
            else:
                trend -= 1
            counter += 1
    return trend


def calculate_trend_for_dfs(dfs, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["days_in_a_row_for_trend_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_trend(temp, row), axis=1
        )


def add_previous_trend(trend_col, row, trend):
    i = row["row_number"]
    if str(trend) == "nan":
        print("NONE")
        return np.nan
    location = int(i - abs(trend))

    if location < 0:
        return np.nan
    return trend_col.iloc[location]


def add_previous_trend_for_dfs(dfs, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        trend_col = temp["days_in_a_row_for_trend_" + dfs_names[i]]
        temp["row_number"] = range(len(temp))

        temp["previous_days_in_a_row_for_trend_" + dfs_names[i]] = temp.apply(
            lambda row: add_previous_trend(
                trend_col, row, row["days_in_a_row_for_trend_" + dfs_names[i]]
            ),
            axis=1,
        )


def get_highest_price(df):
    stock_dict = {}
    highest_price = -10
    for i, row in df.iterrows():
        new_record = 0
        date = row["Date"]  # .strftime('%Y-%m-%d')

        if i == 0:
            stock_dict[date] = np.nan
        else:
            if df.iloc[i - 1]["Close"] > highest_price:
                highest_price = df.iloc[i - 1]["Close"]
                new_record = 1
            stock_dict[date] = new_record
    return stock_dict


def calculate_highest_prices_dict(dfs, dfs_names):
    highest_prices_dict = {}
    for i in range(len(dfs)):
        temp = dfs[i]
        stock_dict = get_highest_price(temp)
        if temp.iloc[-1, :]["Close"] > temp.iloc[:-1, :]["Close"].max():
            new_record = 1
        else:
            new_record = 0
        stock_dict[datetime.today().strftime("%Y-%m-%d")] = new_record
        highest_prices_dict[dfs_names[i]] = stock_dict
    return highest_prices_dict


def add_a_break_record_column(row, records_dict, stock_name):
    date = row["Date"]
    return records_dict[stock_name][date]


def add_a_break_record_column_for_dfs(dfs, highest_prices_dict, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["did_break_highest_all_time_" + dfs_names[i]] = temp.apply(
            lambda row: add_a_break_record_column(
                row, highest_prices_dict, dfs_names[i]
            ),
            axis=1,
        )


def merge_dfs(dfs):
    joined_df = dfs[0]
    for df in dfs[1:]:
        joined_df = pd.concat([joined_df, df.drop(["Date"], axis=1)], axis=1)
    return joined_df


# standard_scaler
def Normalize(Dataframe, standard_scaler=False):
    df_2 = Dataframe.select_dtypes(
        include=["float64", "int64"]
    )  # Create a data with the types that we want
    if not standard_scaler:
        standard_scaler = StandardScaler()  # We initialize our scaler
        standard_scaler.fit(df_2)  # We fit our scaler
        return (
            pd.DataFrame(
                standard_scaler.transform(df_2), index=df_2.index, columns=df_2.columns
            ),
            standard_scaler,
        )  # We transform our data using the scaler we have just fit. (original index)
    else:
        return pd.DataFrame(
            standard_scaler.transform(df_2), index=df_2.index, columns=df_2.columns
        )  # We transforn our test according the train set


def Normalize_minmax(Dataframe, minmax_scaler=False):
    df_2 = Dataframe.select_dtypes(
        include=["float64", "int64"]
    )  # Create a data with the types that we want
    if not minmax_scaler:  # If it's the train data (there is no scaler)
        minmax_scaler = MinMaxScaler()  # We initialize our scaler
        minmax_scaler.fit(df_2)  # We fit our scaler
        return (
            pd.DataFrame(
                minmax_scaler.transform(df_2), index=df_2.index, columns=df_2.columns
            ),
            minmax_scaler,
        )  # We transform our data using the scaler we have just fit. (original index)
    else:
        return pd.DataFrame(
            minmax_scaler.transform(df_2), index=df_2.index, columns=df_2.columns
        )  # We transforn our test according the train set


def after_covid(date):
    return int(date > pd.to_datetime("2019-12-01"))


def pca_cols(relevant_data, threshold, pca=False):
    if not pca:
        pca_num = PCA(n_components=relevant_data.shape[1])  # Initialize PCA object
        pca_num = pca_num.fit(relevant_data)  # Fit the model with the data
        pca_exp = (
            pca_num.explained_variance_ratio_
        )  # Percentage of variance explained by each of the selected components

        print(
            "\n\nFor visualization this is the cumulative explained variance plot for each numeric column"
        )
        plt.plot(
            np.cumsum(pca_exp), color="blueviolet"
        )  # Plot for visualozation- the number of components vs cumulative explained variance
        plt.xlabel("number of components")  # Add title for x-axis
        plt.ylabel("cumulative explained variance")  # Add title for y-axis
        plt.grid()  # Add grid for the plot
        plt.show()  # show the plot

        print(
            "\nNow we apply the pca that will explain at least ", threshold * 100, " %"
        )
        pca = PCA(threshold)  # Initialize PCA object that will explain the threshold
        pca = pca.fit(relevant_data)  # Fit the model with the data
        pca_explain = (
            pca.explained_variance_ratio_
        )  # Percentage of variance explained by each of the selected components
        print(
            "\nThe cumulative explained variance for this pca action is:",
            sum(pca_explain),
        )
        data_after_pca = pd.DataFrame(
            pca.transform(relevant_data), index=relevant_data.index
        )  # Apply the PCA on the train data
        return data_after_pca, pca
    else:
        data_after_pca = pd.DataFrame(
            pca.transform(relevant_data), index=relevant_data.index
        )  # Apply the PCA on the test data
        return data_after_pca


def make_adasyn(x, y):
    ad = ADASYN(random_state=42)  # create adasyn object
    X_res, y_res = ad.fit_resample(x, y)  # fit&resample on adasyn with the data
    return X_res, y_res


def make_smote(x, y):
    sm = SMOTE(sampling_strategy=1, random_state=42)  # create adasyn object
    X_res, y_res = sm.fit_resample(x, y)  # fit&resample on adasyn with the data
    return X_res, y_res


def preproccesing_data(yf_df=False):

    df = create_yf_df()
    tqqq_df = df.iloc[:, [34, 35]]
    sqqq_df = df.iloc[:, [37, 38]]
    uvxy_df = df.iloc[:, [40, 41]]

    # Creating a DataFrame for each stock:
    amzn_df = df.iloc[:, [1, 2, 4]]
    tsla_df = df.iloc[:, [6, 7, 9]]
    msft_df = df.iloc[:, [11, 12, 14]]
    fb_df = df.iloc[:, [16, 17, 19]]
    googl_df = df.iloc[:, [21, 22, 24]]
    aapl_df = df.iloc[:, [26, 27, 29]]
    bac_df = df.iloc[:, [43, 44, 46]]
    ndaq_df = df.iloc[:, [48, 49, 51]]
    tecs_df = df.iloc[:, [53, 54, 56]]

    spy_df = df.iloc[:, [31, 32]]
    #     US_treasury_rate_10_years = df.iloc[:,28]
    dfs = [
        amzn_df,
        tsla_df,
        msft_df,
        fb_df,
        googl_df,
        aapl_df,
        spy_df,
        tqqq_df,
        sqqq_df,
        uvxy_df,
        bac_df,
        ndaq_df,
        tecs_df,
    ]
    dfs_names = [
        "amzn",
        "tsla",
        "msft",
        "fb",
        "googl",
        "aapl",
        "spy",
        "tqqq",
        "sqqq",
        "uvxy",
        "bac",
        "ndaq",
        "tecs",
    ]

    if not yf_df:
        change_date_for_df(dfs)
    calculate_change_in_percentage_for_dfs(dfs, dfs_names)
    calculate_average_volume_for_dfs(dfs, dfs_names)
    calculate_average_close_price_for_dfs(dfs, dfs_names)

    add_label_x_days_before_to_qqq_df(tqqq_df)
    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)
    add_change_for_x_days_to_the_future(tqqq_df)

    add_label_x_days_before_to_qqq_df(sqqq_df, 1)
    add_change_for_x_days_to_the_future(sqqq_df, 1)
    calculate_trend_for_dfs(dfs, dfs_names)
    add_previous_trend_for_dfs(dfs, dfs_names)
    highest_prices_dict = calculate_highest_prices_dict(dfs, dfs_names)
    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, dfs_names)
    merged_dfs = merge_dfs(dfs)
    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()
    #     merged_dfs['10 Years US Treasury Yield Curve Rates'] = merged_dfs['10 Years US Treasury Yield Curve Rates'].astype(float)
    return merged_dfs.iloc[10:-10]


def calculate_how_much_money_i_will_have(
    starting_money,
    threshold,
    clf,
    final_train_data,
    test_data,
    train_lables,
    y_in_percentage_test,
    negative_y_in_percentage_test,
    with_short=False,
):
    beggining_money = starting_money
    end_money = starting_money
    end_money_just_long = starting_money
    clf = XGBClassifier(verbosity=0)

    clf.fit(final_train_data, train_lables)  # Fit the model
    y_proba = clf.predict_proba(test_data)
    y_proba = [v[1] for v in y_proba]
    long_indexes = []
    short_indexes = []
    commission = 1
    money_list = [starting_money]

    just_long_list = [starting_money]
    for i in range(len(y_proba)):
        if y_proba[i] > threshold:
            end_money *= (100 + list(y_in_percentage_test)[i]) / 100
            long_indexes.append(i)
        #             print((100+list(y_in_percentage_test)[i])/100)
        elif with_short:
            short_indexes.append(i)
            end_money *= (100 + list(negative_y_in_percentage_test)[i]) / 100
        else:
            short_indexes.append(i)
        #             print((list(y_in_percentage_test)[i])/100)
        #             print((100-list(y_in_percentage_test)[i])/100)
        money_list.append(end_money)
        end_money_just_long *= (100 + list(y_in_percentage_test)[i]) / 100
        just_long_list.append(end_money_just_long)
    for j in range(len(long_indexes)):
        if j == 0:
            continue
        if long_indexes[j] - long_indexes[j - 1] != 1:
            commission += 1
    if with_short:
        for j in range(len(short_indexes)):
            if j == 0:
                continue
            if short_indexes[j] - short_indexes[j - 1] != 1:
                commission += 1

    print("len of short list is:", len(short_indexes))
    print("len of long list is:", len(long_indexes))
    return (
        end_money,
        end_money / beggining_money,
        short_indexes,
        long_indexes,
        commission,
        money_list,
        just_long_list,
        y_proba,
    )


def plot_lineplot(x, y_model, y_stock):
    x.insert(0, x[0] - timedelta(days=1))
    print("Here is a graph between " + str(x[0])[:10] + " until " + str(x[-1])[:10])
    print("My model- Blue\nTQQQ stock- red")
    plt.plot(x, y_model, label="My money according to the model", color="blue")
    plt.plot(x, y_stock, label="My money if I would invest just in long", color="red")
    plt.legend()
    plt.show()


def show_y_proba_with_change_in_stocks(tqqq_change, sqqq_change, y_proba):
    temp_df = pd.DataFrame(tqqq_change)
    temp_df["sqqq change for 1 day"] = sqqq_change
    temp_df["y_proba"] = y_proba
    return temp_df


def final_split_data_into_train_test(
    all_data,
    features,
    label,
    column_for_real_money_check,
    column_for_real_money_check_2,
):
    x = all_data[
        features + [column_for_real_money_check] + [column_for_real_money_check_2]
    ]
    y = all_data[label]
    x = x.sort_index()
    y = y.sort_index()
    now = datetime.now()
    hour = now.hour
    if hour in [8, 9, 10, 11, 12, 13, 14, 15]:
        until_index = -29
        from_index = -389
    else:
        until_index = -30
        from_index = -390

    x_test = x.iloc[until_index:, :]

    y_test = y.iloc[until_index:, :]
    x_train = x.iloc[from_index:until_index, :]
    y_train = y.iloc[from_index:until_index, :]
    x_train = x_train.sort_index()
    y_train = y_train.sort_index()
    return x_train, x_test, y_train, y_test


def PreProccesing_train_for_final_prediction(
    columns_to_merge_after_normalized,
    columns_to_normalize,
    columns_for_x,
    column_for_y,
    column_for_real_money_check,
    column_for_real_money_check_2,
    new_columns_for_x_temp,
    pca=False,
    adasyn=False,
    yf_df=False,
):
    merged_dfs = preproccesing_data(yf_df)
    x_train_temp, x_test_temp, y_train, y_test = final_split_data_into_train_test(
        merged_dfs,
        columns_for_x,
        column_for_y,
        column_for_real_money_check,
        column_for_real_money_check_2,
    )

    x_train = x_train_temp[columns_for_x]
    x_test = x_test_temp[columns_for_x]

    y_in_percentage_train = x_train_temp[
        column_for_real_money_check
    ]  # .reset_index(drop=True,inplace=True)
    y_in_percentage_test = x_test_temp[
        column_for_real_money_check
    ]  # .reset_index(drop=True,inplace=True)

    negative_y_in_percentage_train = x_train_temp[
        column_for_real_money_check_2
    ]  # .reset_index(drop=True,inplace=True)
    negative_y_in_percentage_test = x_test_temp[
        column_for_real_money_check_2
    ]  # .reset_index(drop=True,inplace=True)

    normal_data_train, standard_scaler = Normalize_minmax(x_train[columns_to_normalize])
    data_after_normalize_train = pd.concat(
        [x_train[columns_to_merge_after_normalized], normal_data_train], axis=1
    )
    x_train_after_normalize = data_after_normalize_train[columns_for_x]
    normal_data_test = Normalize_minmax(x_test[columns_to_normalize], standard_scaler)

    data_after_normalize_test = pd.concat(
        [x_test[columns_to_merge_after_normalized], normal_data_test], axis=1
    )

    x_test_after_normalize = data_after_normalize_test[columns_for_x]
    if adasyn:
        x_train_after_normalize, y_train = make_smote(x_train_after_normalize, y_train)

    x_train_after_normalize = x_train_after_normalize[new_columns_for_x_temp]
    x_test_after_normalize = x_test_after_normalize[new_columns_for_x_temp]
    if pca:

        data_after_pca_train, pca = pca_cols(x_train_after_normalize, 0.75)
        data_after_pca_test = pca_cols(x_test_after_normalize, 0.75, pca)
        return (
            data_after_pca_train,
            data_after_pca_test,
            y_train,
            y_test,
            y_in_percentage_train,
            y_in_percentage_test,
            standard_scaler,
            pca,
        )
    return (
        x_train_after_normalize,
        x_test_after_normalize,
        y_train,
        y_test,
        y_in_percentage_train,
        y_in_percentage_test,
        negative_y_in_percentage_train,
        negative_y_in_percentage_test,
        standard_scaler,
    )


def organized_data_for_prediction(before_midnight, yf_df=False):

    df = create_yf_df()
    new_row = [0] * df.shape[1]
    date_indexes_lst = []
    for i in range(len(df.columns)):
        if df.columns[i] == "Date":
            date_indexes_lst.append(i)
    for j in date_indexes_lst:
        if not before_midnight:
            if (datetime.today().weekday()) not in [5, 6]:
                new_row[j] = pd.to_datetime(
                    datetime.today().strftime("%Y-%m-%d 00:00:00")
                )
            elif (datetime.today().weekday()) == 5:
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
                )
            else:
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
                )
        else:
            if (datetime.today().weekday()) not in [4, 5]:
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
                )
            elif (datetime.today().weekday()) == 4:
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=3)).strftime("%Y-%m-%d 00:00:00")
                )
            else:
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
                )
    df.loc[len(df)] = new_row

    tqqq_df = df.iloc[:, [34, 35]]
    sqqq_df = df.iloc[:, [37, 38]]
    uvxy_df = df.iloc[:, [40, 41]]

    # Creating a DataFrame for each stock:
    amzn_df = df.iloc[:, [1, 2, 4]]
    tsla_df = df.iloc[:, [6, 7, 9]]
    msft_df = df.iloc[:, [11, 12, 14]]
    fb_df = df.iloc[:, [16, 17, 19]]
    googl_df = df.iloc[:, [21, 22, 24]]
    aapl_df = df.iloc[:, [26, 27, 29]]
    bac_df = df.iloc[:, [43, 44, 46]]
    ndaq_df = df.iloc[:, [48, 49, 51]]
    tecs_df = df.iloc[:, [53, 54, 56]]

    spy_df = df.iloc[:, [31, 32]]
    #     US_treasury_rate_10_years = df.iloc[:,28]
    dfs = [
        amzn_df,
        tsla_df,
        msft_df,
        fb_df,
        googl_df,
        aapl_df,
        spy_df,
        tqqq_df,
        sqqq_df,
        uvxy_df,
        bac_df,
        ndaq_df,
        tecs_df,
    ]
    dfs_names = [
        "amzn",
        "tsla",
        "msft",
        "fb",
        "googl",
        "aapl",
        "spy",
        "tqqq",
        "sqqq",
        "uvxy",
        "bac",
        "ndaq",
        "tecs",
    ]
    if not yf_df:
        change_date_for_df(dfs)
    calculate_change_in_percentage_for_dfs(dfs, dfs_names)
    calculate_average_volume_for_dfs(dfs, dfs_names)
    calculate_average_close_price_for_dfs(dfs, dfs_names)

    add_label_x_days_before_to_qqq_df(tqqq_df)
    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)
    add_change_for_x_days_to_the_future(tqqq_df)

    add_change_for_x_days_to_the_future(sqqq_df, 1)
    add_label_x_days_before_to_qqq_df(sqqq_df, 1)

    calculate_trend_for_dfs(dfs, dfs_names)
    add_previous_trend_for_dfs(dfs, dfs_names)

    highest_prices_dict = calculate_highest_prices_dict(dfs, dfs_names)
    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, dfs_names)
    merged_dfs = merge_dfs(dfs)
    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()
    #     new_row = pd.DataFrame([[datetime.today().strftime("%Y-%m-%d")]],columns = ['Date','Close','Volume'])

    #     merged_dfs['10 Years US Treasury Yield Curve Rates'] = merged_dfs['10 Years US Treasury Yield Curve Rates'].astype(float)
    return merged_dfs


def update_merge_df(previous_merge, before_midnight, yf_df=False):
    df = create_yf_df()
    df = df.iloc[-10:, :].reset_index(drop=True)

    new_row = [0] * df.shape[1]
    date_indexes_lst = []
    for i in range(len(df.columns)):
        if df.columns[i] == "Date":
            date_indexes_lst.append(i)
    for j in date_indexes_lst:
        if not before_midnight:
            if (datetime.today().weekday()) not in [5, 6]:
                print("a")
                new_row[j] = pd.to_datetime(
                    datetime.today().strftime("%Y-%m-%d 00:00:00")
                )
            elif (datetime.today().weekday()) == 5:
                print("b")
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
                )
            else:
                print("c")
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
                )
        else:
            if (datetime.today().weekday()) not in [4, 5]:
                print("d")
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
                )
            elif (datetime.today().weekday()) == 4:
                print("e")
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=3)).strftime("%Y-%m-%d 00:00:00")
                )
            else:
                print("f")
                new_row[j] = pd.to_datetime(
                    (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
                )
    df.loc[len(df)] = new_row
    tqqq_df = df.iloc[:, [34, 35]]
    sqqq_df = df.iloc[:, [37, 38]]
    uvxy_df = df.iloc[:, [40, 41]]

    # Creating a DataFrame for each stock:
    amzn_df = df.iloc[:, [1, 2, 4]]
    tsla_df = df.iloc[:, [6, 7, 9]]
    msft_df = df.iloc[:, [11, 12, 14]]
    fb_df = df.iloc[:, [16, 17, 19]]
    googl_df = df.iloc[:, [21, 22, 24]]
    aapl_df = df.iloc[:, [26, 27, 29]]
    bac_df = df.iloc[:, [43, 44, 46]]
    ndaq_df = df.iloc[:, [48, 49, 51]]
    tecs_df = df.iloc[:, [53, 54, 56]]

    spy_df = df.iloc[:, [31, 32]]
    #     US_treasury_rate_10_years = df.iloc[:,28]
    dfs = [
        amzn_df,
        tsla_df,
        msft_df,
        fb_df,
        googl_df,
        aapl_df,
        spy_df,
        tqqq_df,
        sqqq_df,
        uvxy_df,
        bac_df,
        ndaq_df,
        tecs_df,
    ]
    dfs_names = [
        "amzn",
        "tsla",
        "msft",
        "fb",
        "googl",
        "aapl",
        "spy",
        "tqqq",
        "sqqq",
        "uvxy",
        "bac",
        "ndaq",
        "tecs",
    ]
    if not yf_df:
        change_date_for_df(dfs)

    calculate_change_in_percentage_for_dfs(dfs, dfs_names)

    calculate_average_volume_for_dfs(dfs, dfs_names)

    calculate_average_close_price_for_dfs(dfs, dfs_names)

    add_label_x_days_before_to_qqq_df(tqqq_df)

    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)

    add_change_for_x_days_to_the_future(tqqq_df)

    add_change_for_x_days_to_the_future(sqqq_df, 1)

    add_label_x_days_before_to_qqq_df(sqqq_df, 1)

    calculate_trend_for_dfs(dfs, dfs_names)

    add_previous_trend_for_dfs(dfs, dfs_names)

    highest_prices_dict = calculate_highest_prices_dict(dfs, dfs_names)

    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, dfs_names)

    merged_dfs = merge_dfs(dfs)

    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()

    new_merge = previous_merge.copy()
    new_merge.iloc[-1, :] = merged_dfs.iloc[-1, :]

    return pd.DataFrame(new_merge)


def send_proba(clf, train_data, test_row, train_labels, threshold):
    clf.fit(train_data, train_labels)  # Fit the model
    y_proba = clf.predict_proba(test_row)
    y_proba = y_proba[0][1]
    if y_proba > threshold:
        print(y_proba, "buy")
        return y_proba, "buy"
    else:
        print(y_proba, "sell")
        return y_proba, "sell"


def predict_today(
    df,
    columns_for_x,
    column_for_real_money_check,
    column_for_real_money_check2,
    standard_scaler,
    columns_to_normalize,
    columns_to_merge_after_normalized,
    new_columns_for_x_temp,
    pca=False,
):
    x_test = df[columns_for_x]
    y_in_percentage_test = df[
        column_for_real_money_check
    ]  # .reset_index(drop=True,inplace=True)
    negative_y_in_percentage_test = df[
        column_for_real_money_check2
    ]  # .reset_index(drop=True,inplace=True)

    normal_data_test = Normalize_minmax(x_test[columns_to_normalize], standard_scaler)
    data_after_normalize_test = pd.concat(
        [x_test[columns_to_merge_after_normalized], normal_data_test], axis=1
    )
    x_test_after_normalize = data_after_normalize_test[columns_for_x]

    x_test_after_normalize = x_test_after_normalize[new_columns_for_x_temp]
    if pca:
        data_after_pca_test = pca_cols(x_test_after_normalize, 0.75, pca)
        return data_after_pca_test, y_in_percentage_test
    return x_test_after_normalize, y_in_percentage_test, negative_y_in_percentage_test


def predict_todays_direction(
    clf,
    train_data,
    train_labels,
    threshold,
    previous_merge,
    columns_for_x,
    positive_column_for_real_money_check,
    negative_column_for_real_money_check,
    final_standard_scaler,
    columns_to_normalize,
    columns_to_merge_after_normalized,
    new_columns_for_x_temp,
    no_trading_dates,
):
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    if hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        before_midnight = False
    else:
        before_midnight = True
    merged_dfs_new = update_merge_df(previous_merge, before_midnight, yf_df=True)
    x_test, y_test, negative_y_test = predict_today(
        merged_dfs_new,
        columns_for_x,
        positive_column_for_real_money_check,
        negative_column_for_real_money_check,
        final_standard_scaler,
        columns_to_normalize,
        columns_to_merge_after_normalized,
        new_columns_for_x_temp,
        pca=False,
    )

    todays_row = pd.DataFrame(x_test.iloc[-1]).T

    while True:
        if todays_row.index in no_trading_dates:
            todays_row.index = todays_row.index + timedelta(days=1)
            todays_row["Weekday"] += 1
            if list(todays_row["Weekday"])[0] == 5:
                todays_row.index = todays_row.index + timedelta(days=2)
                todays_row["Weekday"] = 0
            elif list(todays_row["Weekday"])[0] == 6:
                todays_row.index = todays_row.index + timedelta(days=1)
                todays_row["Weekday"] = 0
        else:
            break
    proba, action = send_proba(clf, train_data, todays_row, train_labels, threshold)
    todays_row = todays_row.fillna(0)
    return proba, action, todays_row


def second_predict_todays_direction(
    clf, train_data, test_row, train_labels, threshold=0.15
):
    clf.fit(train_data, train_labels)  # Fit the model
    y_proba = clf.predict_proba(test_row)
    y_proba = y_proba[0][1]
    if threshold == 0.15:
        if y_proba > 0.95:
            print(y_proba, "strong buy")
            return y_proba, "buy"
        elif y_proba > threshold:
            print(y_proba, " buy")
            return y_proba, "buy"
        elif y_proba < 0.05:
            print(y_proba, "strong sell")
            return y_proba, "sell"
        else:
            print(y_proba, "sell")
            return y_proba, "sell"
    else:
        if y_proba > threshold:
            print(y_proba, "buy")
            return y_proba, "buy"
        else:
            print(y_proba, "sell")
            return y_proba, "sell"
