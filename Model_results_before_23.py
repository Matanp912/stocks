from stocks.utils import (
    PreProccesing_train_for_final_prediction,
    organized_data_for_prediction,
    predict_todays_direction,
    COLUMNS_TO_MERGE_AFTER_NORMALIZED,
    COLUMNS_TO_NORMALIZE,
    COLUMNS_FOR_X,
    COLUMN_FOR_Y,
    POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
    NEW_COLUMN_FOR_X_TEMP,
    XGB_CLF,
    GROUP_NAME,
    PRIVATE_GROUP_NAME,
    NO_TRADING_DAYS,
    WHATSAPP_ID,
    PHONE_NUMBER
)

import time
from datetime import datetime
import pywhatkit as pwt


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

now = datetime.now()
hour = now.hour
minute = now.minute
if hour in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    before_midnight = False
else:
    before_midnight = True

merged_dfs = organized_data_for_prediction(before_midnight, yf_df=True)

print("Done First stage")
print("last_month_test_data")
print(last_month_test_data)
print(final_train_data_for_prediction)
print("final_train_data_for_prediction.shape", final_train_data_for_prediction.shape)
while datetime.now().hour < 23:
    now = datetime.now()
    hour = now.hour
    minute = now.minute

    if hour == 17 and minute == 30:
        pwt.open_whatsapp(WHATSAPP_ID)
        time.sleep(60)
    if hour == 17 and minute == 40:
        proba, action, todays_row = predict_todays_direction(
            XGB_CLF,
            final_train_data_for_prediction,
            final_train_lables_for_prediction,
            0.15,
            merged_dfs,
            COLUMNS_FOR_X,
            POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            final_standard_scaler,
            COLUMNS_TO_NORMALIZE,
            COLUMNS_TO_MERGE_AFTER_NORMALIZED,
            NEW_COLUMN_FOR_X_TEMP,
            NO_TRADING_DAYS,
        )
        pwt.manual_send_message_to_someone(
            PRIVATE_GROUP_NAME, action + " " + str(proba)
        )
        time.sleep(60)

    if hour == 20 and minute == 30:
        pwt.open_whatsapp(WHATSAPP_ID)
        time.sleep(60)
    if hour == 20 and minute == 40:
        proba, action, todays_row = predict_todays_direction(
            XGB_CLF,
            final_train_data_for_prediction,
            final_train_lables_for_prediction,
            0.15,
            merged_dfs,
            COLUMNS_FOR_X,
            POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            final_standard_scaler,
            COLUMNS_TO_NORMALIZE,
            COLUMNS_TO_MERGE_AFTER_NORMALIZED,
            NEW_COLUMN_FOR_X_TEMP,
            NO_TRADING_DAYS,
        )
        pwt.manual_send_message_to_someone(
            PRIVATE_GROUP_NAME, action + " " + str(proba)
        )
        time.sleep(60)

    if hour == 22 and minute == 48:
        print("Alive! Opening Whatsapp")
        pwt.open_whatsapp(WHATSAPP_ID)
        time.sleep(60)
    if hour == 22 and minute == 55:
        proba, action, todays_row = predict_todays_direction(
            XGB_CLF,
            final_train_data_for_prediction,
            final_train_lables_for_prediction,
            0.15,
            merged_dfs,
            COLUMNS_FOR_X,
            POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
            final_standard_scaler,
            COLUMNS_TO_NORMALIZE,
            COLUMNS_TO_MERGE_AFTER_NORMALIZED,
            NEW_COLUMN_FOR_X_TEMP,
            NO_TRADING_DAYS,
        )
        pwt.manual_send_message_to_someone(
            PRIVATE_GROUP_NAME, action + " " + str(proba)
        )
        time.sleep(30)
        break


while True:
    proba, action, todays_row = predict_todays_direction(
        XGB_CLF,
        final_train_data_for_prediction,
        final_train_lables_for_prediction,
        0.15,
        merged_dfs,
        COLUMNS_FOR_X,
        POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        final_standard_scaler,
        COLUMNS_TO_NORMALIZE,
        COLUMNS_TO_MERGE_AFTER_NORMALIZED,
        NEW_COLUMN_FOR_X_TEMP,
        NO_TRADING_DAYS,
    )
    print("Sending message. The time is: ", str(datetime.now()))
    print(str(proba))
    pwt.manual_send_message_to_someone(GROUP_NAME, action + " " + str(proba))
    time.sleep(30)
    pwt.sendwhatmsg_instantly_to_a_number(PHONE_NUMBER, action + " " + str(proba))
    time.sleep(20)

    pwt.manual_send_message_to_someone(PRIVATE_GROUP_NAME, str(todays_row))
    break

while datetime.now().hour < 23:
    time.sleep(30)
time.sleep(120)
while True:
    proba, action, todays_row = predict_todays_direction(
        XGB_CLF,
        final_train_data_for_prediction,
        final_train_lables_for_prediction,
        0.15,
        merged_dfs,
        COLUMNS_FOR_X,
        POSITIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        NEGATIVE_COLUMN_FOR_REAL_MONEY_CHECK,
        final_standard_scaler,
        COLUMNS_TO_NORMALIZE,
        COLUMNS_TO_MERGE_AFTER_NORMALIZED,
        NEW_COLUMN_FOR_X_TEMP,
        NO_TRADING_DAYS,
    )
    print("Sending message. The time is: ", str(datetime.now()))
    pwt.sendwhatmsg_instantly_to_a_number(PHONE_NUMBER, action + " " + str(proba))
    time.sleep(30)
    pwt.manual_send_message_to_someone(GROUP_NAME, action + " " + str(proba))
    time.sleep(30)

    pwt.manual_send_message_to_someone(PRIVATE_GROUP_NAME, str(todays_row))
    break
