from functions_and_parameters import (
    PreProccesing_train_for_final_prediction,
    organized_data_for_prediction,
    predict_today,
    second_predict_todays_direction,
    columns_for_x,
    columns_to_normalize,
    columns_to_merge_after_normalized,
    column_for_y,
    positive_column_for_real_money_check,
    negative_column_for_real_money_check,
    new_columns_for_x_temp,
    remove_lst,
    WHATSAPP_ID,
    PHONE_NUMBER,
)
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import pywhatkit as pwt

from xgboost import XGBClassifier

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")


for i in remove_lst:
    new_columns_for_x_temp.remove(i)
XGB_clf = XGBClassifier(verbosity=0)
group_name = "Test_group"


def main(
    final_train_data_for_prediction,
    final_train_lables_for_prediction,
    final_standard_scaler,
):
    now = datetime.now()
    hour = now.hour
    if hour == 23:
        now = datetime.now()
        hour = now.hour
        pwt.manual_send_message_to_someone(group_name, "Making the final decision...")
        if hour != 23:
            before_midnight = False
        else:
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
        no_trading_dates = ["2021-09-06", "2021-11-25", "2021-12-24"]

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
        print(todays_row)
        proba, action = second_predict_todays_direction(
            XGB_clf,
            final_train_data_for_prediction,
            todays_row,
            final_train_lables_for_prediction,
            threshold=0.15,
        )
        return action, proba, todays_row
    else:
        return False, False, False


run_me = False
if run_me:
    while datetime.now().hour < 23:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        if hour == 22 and minute == 54:
            print("Alive! Opening Whatsapp")
            pwt.open_whatsapp(WHATSAPP_ID)
            time.sleep(60)

    (
        final_train_data_for_prediction,
        last_month_test_data,
        final_train_lables_for_prediction,
        last_month_label,
        final_y_in_percentage_train,
        final_y_in_percentage_last_month,
        final_negative_y_in_percentage_train,
        final_negative_y_in_percentage_last_month,
        final_standard_scaler,
    ) = PreProccesing_train_for_final_prediction(
        columns_to_merge_after_normalized,
        columns_to_normalize,
        columns_for_x,
        column_for_y,
        positive_column_for_real_money_check,
        negative_column_for_real_money_check,
        new_columns_for_x_temp,
        pca=False,
        adasyn=False,
        yf_df=True,
    )
    print("Done First stage")
    print(final_train_data_for_prediction)
    print(final_train_data_for_prediction.shape)
    while True:
        action, proba, todays_row = main(
            final_train_data_for_prediction,
            final_train_lables_for_prediction,
            final_standard_scaler,
        )
        if action:
            print("Sending message. The time is: ", str(datetime.now()))
            pwt.sendwhatmsg_instantly(WHATSAPP_ID, action)
            time.sleep(30)
            pwt.sendwhatmsg_instantly(WHATSAPP_ID, str(proba))
            pwt.sendwhatmsg_instantly(WHATSAPP_ID, str(todays_row))
            break
        else:

            print("too early, trying again. The time is: ", str(datetime.now()))
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            while hour < 22:
                now = datetime.now()
                if now.hour != hour:
                    hour = now.hour

                if now.minute != minute:
                    minute = now.minute
                    print(
                        str(22 - hour - 1)
                        + " hours and "
                        + str(60 - minute)
                        + " minutes left"
                    )
                if hour == 22 and minute == 54:
                    pwt.open_whatsapp(WHATSAPP_ID)
                    time.sleep(60)
