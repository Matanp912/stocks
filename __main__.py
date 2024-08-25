import time
import pandas as pd
from datetime import datetime, timedelta
import pywhatkit as pwt


from utils import (
    PreProccesing_train_for_final_prediction,
    organized_data_for_prediction,
    predict_today,
    second_predict_todays_direction,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from params import (
    COLUMNS_TO_MERGE_AFTER_NORMALIZED,
    COLUMNS_TO_NORMALIZE,
    COLUMNS_FOR_X,
    POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEW_COLUMN_FOR_X_TEMP,
    XGB_CLF,
    GROUP_NAME,
    COLUMN_FOR_Y,
    WHATSAPP_ID,
)


def main(
    final_train_data_for_prediction: pd.DataFrame,
    final_train_lables_for_prediction: pd.Series,
    final_standard_scaler: StandardScaler | MinMaxScaler,
):
    now = datetime.now()
    hour = now.hour
    if hour == 23:
        now = datetime.now()
        hour = now.hour
        pwt.manual_send_message_to_someone(GROUP_NAME, "Making the final decision...")
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
        NO_TRADING_DAYS = ["2021-09-06", "2021-11-25", "2021-12-24"]

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
        print(todays_row)
        proba, action = second_predict_todays_direction(
            XGB_CLF,
            final_train_data_for_prediction,
            todays_row,
            final_train_lables_for_prediction,
            threshold=0.15,
        )
        return action, proba, todays_row
    else:
        return False, False, False


if __name__ == "__main__":
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
        COLUMNS_TO_MERGE_AFTER_NORMALIZED,
        COLUMNS_TO_NORMALIZE,
        COLUMNS_FOR_X,
        COLUMN_FOR_Y,
        POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        NEW_COLUMN_FOR_X_TEMP,
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
