import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pywhatkit as pwt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from imblearn.over_sampling import ADASYN, SMOTE

from params import (
    COLUMNS_TO_MERGE_AFTER_NORMALIZED,
    COLUMNS_TO_NORMALIZE,
    COLUMNS_FOR_X,
    POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEW_COLUMN_FOR_X_TEMP,
    XGB_CLF,
    GROUP_NAME,
    NO_TRADING_DAYS,
    YF_DF_COLS,
    YF_DF_RENAME,
    DFS_NAMES,
    CLOSE_DATA_DICT,
    VOLUME_DATA_DICT,
)

pd.set_option("display.max_columns", None)


def main_before_23(
    final_train_data_for_prediction: pd.DataFrame,
    final_train_lables_for_prediction: pd.Series,
    final_standard_scaler: StandardScaler | MinMaxScaler,
    public: bool = True,
) -> tuple[str, float, pd.Series]:
    if public:
        pwt.manual_send_message_to_someone(GROUP_NAME, "Making the first decision...")

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
        COLUMNS_FOR_X,
        POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        final_standard_scaler,
        COLUMNS_TO_NORMALIZE,
        COLUMNS_TO_MERGE_AFTER_NORMALIZED,
        NEW_COLUMN_FOR_X_TEMP,
        pca=False,
    )

    todays_row = pd.DataFrame(x_test.iloc[-1]).T

    while True:
        if todays_row.index in NO_TRADING_DAYS:
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
        XGB_CLF,
        final_train_data_for_prediction,
        todays_row,
        final_train_lables_for_prediction,
        threshold=0.15,
    )

    return action, proba, todays_row


def create_yf_df() -> pd.DataFrame:
    today = str(datetime.today() + timedelta(days=1)).split(" ")[0]
    data = yf.download(
        "AMZN TSLA MSFT FB GOOGL AAPL SPY TQQQ SQQQ UVXY BAC NDAQ TECS",
        start="2012-06-01",
        end=today,
    )
    if (data.iloc[-1, 25] / data.iloc[-2, 25]) < 0.7:
        data.iloc[-1:, 25] *= 2
        data.iloc[-1:, 23] /= 5

    close_data = data["Close"].rename(columns=CLOSE_DATA_DICT)
    Volume_data = data["Volume"].rename(columns=VOLUME_DATA_DICT)

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

    yf_df = yf_df[YF_DF_COLS].rename(columns=YF_DF_RENAME)
    return yf_df


def change_date(date: str) -> pd.Timestamp:
    day, month, year = date[0:10].split("/")
    return pd.to_datetime(year + month + day)


def change_date_for_df(dfs: list[pd.DataFrame]):
    for df in dfs:
        df["Date"] = df["Date"].apply(change_date)
        if df.shape[1] == 3:
            df["Volume"] = df["Volume"].astype(float)
        df["Close"] = df["Close"].astype(float)


def calculate_change_in_percentage(
    df: pd.DataFrame, days: int, row: pd.Series
) -> float:
    i = row.name
    if i - days > 0:
        current_price = df.iloc[i - 1, :]["Close"]
        past_price = df.iloc[i - 1 - days]["Close"]

        return (current_price / past_price - 1) * 100
    else:
        return np.nan


def calculate_change_in_percentage_for_dfs(
    dfs: list[pd.DataFrame], dfs_names: list[str]
):
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


def calculate_average_volume(df: pd.DataFrame, days: int, row: pd.Series) -> float:

    i = row.name
    if i - days > 0:
        mean_volume = df.iloc[i - days : i, :]["Volume"].mean()

        return mean_volume
    else:
        return np.nan


def calculate_average_volume_for_dfs(dfs: list[pd.DataFrame], dfs_names: list[str]):
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


def calculate_average_close_price(df: pd.DataFrame, days: int, row: pd.Series) -> float:

    i = row.name
    if i - days > 0:
        mean_price = df.iloc[i - days : i, :]["Close"].mean()

        return mean_price
    else:
        return np.nan


def calculate_average_close_price_for_dfs(dfs: list[pd.DataFrame], dfs_names):
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


def add_label_x_days_before(df: pd.DataFrame, days: int, row: pd.Series) -> float:
    i = row.name
    if i - days > 0:
        current_price = df.iloc[i - 1, :]["Close"]
        past_price = df.iloc[i - 1 - days]["Close"]

        return int(current_price > past_price)
    else:
        return np.nan


def add_label_x_days_before_to_qqq_df(qqq_df: pd.DataFrame, sqqq: bool = False):
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


def add_label_x_days_to_the_future(
    df: pd.DataFrame, days: int, row: pd.Series
) -> float:
    i = row.name
    if i - 1 < 0:
        return np.nan
    if i + days <= df.shape[0]:
        current_price = df.iloc[i - 1, :]["Close"]
        future_price = df.iloc[i - 1 + days]["Close"]
        return int(future_price > current_price)
    else:
        return np.nan


def add_label_x_days_to_the_future_to_qqq_df(qqq_df: pd.DataFrame):
    qqq_df["label_1_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 1, row), axis=1
    )
    qqq_df["label_7_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 7, row), axis=1
    )
    qqq_df["label_30_day"] = qqq_df.apply(
        lambda row: add_label_x_days_to_the_future(qqq_df, 30, row), axis=1
    )


def calculate_change_for_x_days_to_the_future(
    df: pd.DataFrame, days: int, row: pd.Series
) -> float:
    i = row.name
    if i - 1 < 0:
        return np.nan
    if i + days <= df.shape[0]:
        current_price = df.iloc[i - 1, :]["Close"]
        future_price = df.iloc[i - 1 + days]["Close"]
        return (future_price / current_price - 1) * 100
    else:
        return np.nan


def add_change_for_x_days_to_the_future(qqq_df: pd.DataFrame, sqqq: bool = False):
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


def calculate_trend(df: pd.DataFrame, row: pd.Series) -> float:
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


def calculate_trend_for_dfs(dfs: pd.DataFrame, dfs_names):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["days_in_a_row_for_trend_" + dfs_names[i]] = temp.apply(
            lambda row: calculate_trend(temp, row), axis=1
        )


def add_previous_trend(trend_col: pd.Series, row: pd.Series, trend: float) -> float:
    i = row["row_number"]
    if str(trend) == "nan":
        print("NONE")
        return np.nan
    location = int(i - abs(trend))

    if location < 0:
        return np.nan
    return trend_col.iloc[location]


def add_previous_trend_for_dfs(dfs: list[pd.DataFrame], dfs_names: list[str]):
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


def get_highest_price(df: pd.DataFrame) -> dict[pd.Timestamp, float]:
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


def calculate_highest_prices_dict(
    dfs: list[pd.DataFrame], dfs_names: list[str]
) -> dict[str, dict[pd.Timestamp, float]]:
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


def add_a_break_record_column(
    row: pd.Series, records_dict: dict[str, dict[pd.Timestamp, float]], stock_name: str
) -> float:
    date = row["Date"]
    return records_dict[stock_name][date]


def add_a_break_record_column_for_dfs(
    dfs: list[pd.DataFrame],
    highest_prices_dict: dict[str, dict[pd.Timestamp, float]],
    dfs_names: list[str],
):
    for i in range(len(dfs)):
        temp = dfs[i]
        temp["did_break_highest_all_time_" + dfs_names[i]] = temp.apply(
            lambda row: add_a_break_record_column(
                row, highest_prices_dict, dfs_names[i]
            ),
            axis=1,
        )


def merge_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    joined_df = dfs[0]
    for df in dfs[1:]:
        joined_df = pd.concat([joined_df, df.drop(["Date"], axis=1)], axis=1)
    return joined_df


# standard_scaler
def Normalize(df: pd.DataFrame, standard_scaler: bool = False) -> pd.DataFrame:
    df_2 = df.select_dtypes(
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


def Normalize_minmax(df: pd.DataFrame, minmax_scaler: bool = False) -> pd.DataFrame:
    df_2 = df.select_dtypes(
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


def after_covid(date: pd.Timestamp) -> int:
    return int(date > pd.to_datetime("2019-12-01"))


def pca_cols(relevant_data: pd.DataFrame, threshold: float, pca: bool = False):
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


def make_adasyn(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    ad = ADASYN(random_state=42)  # create adasyn object
    X_res, y_res = ad.fit_resample(x, y)  # fit&resample on adasyn with the data
    return X_res, y_res


def make_smote(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    sm = SMOTE(sampling_strategy=1, random_state=42)  # create adasyn object
    X_res, y_res = sm.fit_resample(x, y)  # fit&resample on adasyn with the data
    return X_res, y_res


def preproccesing_data(yf_df: bool = False) -> pd.DataFrame:

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

    if not yf_df:
        change_date_for_df(dfs)
    calculate_change_in_percentage_for_dfs(dfs, DFS_NAMES)
    calculate_average_volume_for_dfs(dfs, DFS_NAMES)
    calculate_average_close_price_for_dfs(dfs, DFS_NAMES)

    add_label_x_days_before_to_qqq_df(tqqq_df)
    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)
    add_change_for_x_days_to_the_future(tqqq_df)

    add_label_x_days_before_to_qqq_df(sqqq_df, 1)
    add_change_for_x_days_to_the_future(sqqq_df, 1)
    calculate_trend_for_dfs(dfs, DFS_NAMES)
    add_previous_trend_for_dfs(dfs, DFS_NAMES)
    highest_prices_dict = calculate_highest_prices_dict(dfs, DFS_NAMES)
    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, DFS_NAMES)
    merged_dfs = merge_dfs(dfs)
    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()
    #     merged_dfs['10 Years US Treasury Yield Curve Rates'] = merged_dfs['10 Years US Treasury Yield Curve Rates'].astype(float)
    return merged_dfs.iloc[10:-10]


def calculate_how_much_money_i_will_have(
    starting_money: float,
    threshold: float,
    clf,
    final_train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_lables: pd.Series,
    y_in_percentage_test: pd.Series,
    negative_y_in_percentage_test: pd.Series,
    with_short: bool = False,
):
    beggining_money = starting_money
    end_money = starting_money
    end_money_just_long = starting_money
    clf = XGB_CLF

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


def show_y_proba_with_change_in_stocks(
    tqqq_change: pd.Series, sqqq_change: pd.Series, y_proba: pd.Series
) -> pd.DataFrame:
    temp_df = pd.DataFrame(tqqq_change)
    temp_df["sqqq change for 1 day"] = sqqq_change
    temp_df["y_proba"] = y_proba
    return temp_df


def final_split_data_into_train_test(
    all_data: pd.DataFrame,
    features: pd.DataFrame,
    label: pd.Series,
    column_for_real_money_check: str,
    column_for_real_money_check_2: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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
    columns_to_merge_after_normalized: list[str],
    columns_to_normalize: list[str],
    columns_for_x: list[str],
    columns_for_y: list[str],
    column_for_real_money_check: str,
    column_for_real_money_check_2: str,
    new_column_for_x_temp: str,
    pca: bool = False,
    adasyn: bool = False,
    yf_df: bool = False,
):
    merged_dfs = preproccesing_data(yf_df)
    x_train_temp, x_test_temp, y_train, y_test = final_split_data_into_train_test(
        merged_dfs,
        columns_for_x,
        columns_for_y,
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

    x_train_after_normalize = x_train_after_normalize[new_column_for_x_temp]
    x_test_after_normalize = x_test_after_normalize[new_column_for_x_temp]
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


def organized_data_for_prediction(before_midnight: bool, yf_df: bool = False):

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

    if not yf_df:
        change_date_for_df(dfs)
    calculate_change_in_percentage_for_dfs(dfs, DFS_NAMES)
    calculate_average_volume_for_dfs(dfs, DFS_NAMES)
    calculate_average_close_price_for_dfs(dfs, DFS_NAMES)

    add_label_x_days_before_to_qqq_df(tqqq_df)
    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)
    add_change_for_x_days_to_the_future(tqqq_df)

    add_change_for_x_days_to_the_future(sqqq_df, 1)
    add_label_x_days_before_to_qqq_df(sqqq_df, 1)

    calculate_trend_for_dfs(dfs, DFS_NAMES)
    add_previous_trend_for_dfs(dfs, DFS_NAMES)

    highest_prices_dict = calculate_highest_prices_dict(dfs, DFS_NAMES)
    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, DFS_NAMES)
    merged_dfs = merge_dfs(dfs)
    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()
    #     new_row = pd.DataFrame([[datetime.today().strftime("%Y-%m-%d")]],columns = ['Date','Close','Volume'])

    #     merged_dfs['10 Years US Treasury Yield Curve Rates'] = merged_dfs['10 Years US Treasury Yield Curve Rates'].astype(float)
    return merged_dfs


def update_merge_df(
    previous_merge: pd.DataFrame, before_midnight: bool, yf_df: bool = False
) -> pd.DataFrame:
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

    if not yf_df:
        change_date_for_df(dfs)

    calculate_change_in_percentage_for_dfs(dfs, DFS_NAMES)

    calculate_average_volume_for_dfs(dfs, DFS_NAMES)

    calculate_average_close_price_for_dfs(dfs, DFS_NAMES)

    add_label_x_days_before_to_qqq_df(tqqq_df)

    add_label_x_days_to_the_future_to_qqq_df(tqqq_df)

    add_change_for_x_days_to_the_future(tqqq_df)

    add_change_for_x_days_to_the_future(sqqq_df, 1)

    add_label_x_days_before_to_qqq_df(sqqq_df, 1)

    calculate_trend_for_dfs(dfs, DFS_NAMES)

    add_previous_trend_for_dfs(dfs, DFS_NAMES)

    highest_prices_dict = calculate_highest_prices_dict(dfs, DFS_NAMES)

    add_a_break_record_column_for_dfs(dfs, highest_prices_dict, DFS_NAMES)

    merged_dfs = merge_dfs(dfs)

    merged_dfs = merged_dfs.drop(["Volume", "Close"], axis=1)
    merged_dfs["After Covid"] = merged_dfs["Date"].apply(after_covid)
    #     merged_dfs = pd.concat([merged_dfs,US_treasury_rate_10_years],axis=1)
    merged_dfs["Weekday"] = merged_dfs["Date"].dt.dayofweek

    merged_dfs = merged_dfs.set_index("Date").sort_index()

    new_merge = previous_merge.copy()
    new_merge.iloc[-1, :] = merged_dfs.iloc[-1, :]

    return pd.DataFrame(new_merge)


def send_proba(
    clf,
    train_data: pd.DataFrame,
    test_row: pd.Series,
    train_labels: pd.Series,
    threshold: float,
) -> tuple[float, str]:
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
    df: pd.DataFrame,
    columns_for_x: list[str],
    column_for_real_money_check: str,
    column_for_real_money_check2: str,
    standard_scaler: StandardScaler | MinMaxScaler,
    columns_to_normalize: list[str],
    columns_to_merge_after_normalized: list[str],
    new_column_for_x_temp: str,
    pca: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
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

    x_test_after_normalize = x_test_after_normalize[new_column_for_x_temp]
    if pca:
        data_after_pca_test = pca_cols(x_test_after_normalize, 0.75, pca)
        return data_after_pca_test, y_in_percentage_test
    return x_test_after_normalize, y_in_percentage_test, negative_y_in_percentage_test


def predict_todays_direction(
    clf,
    train_data: pd.DataFrame,
    train_labels: pd.Series,
    threshold: float,
    previous_merge: pd.DataFrame,
    columns_for_x: list[str],
    positive_column_for_real_money_check: str,
    negative_column_for_real_money_check: str,
    final_standard_scaler: StandardScaler | MinMaxScaler,
    columns_to_normalize: list[str],
    columns_to_merge_after_normalized: list[str],
    new_column_for_x_temp: str,
    no_trading_days: list[str],
) -> tuple[float, str, pd.Series]:
    now = datetime.now()
    hour = now.hour
    minute = now.minute
    if hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        before_midnight = False
    else:
        before_midnight = True
    merged_dfs_new = update_merge_df(previous_merge, before_midnight, yf_df=True)
    x_test, _, _ = predict_today(
        merged_dfs_new,
        columns_for_x,
        positive_column_for_real_money_check,
        negative_column_for_real_money_check,
        final_standard_scaler,
        columns_to_normalize,
        columns_to_merge_after_normalized,
        new_column_for_x_temp,
        pca=False,
    )

    todays_row = pd.DataFrame(x_test.iloc[-1]).T

    while True:
        if todays_row.index in no_trading_days:
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
    clf,
    train_data: pd.DataFrame,
    test_row: pd.Series,
    train_labels: pd.Series,
    threshold: float = 0.15,
) -> tuple[float, str]:
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
