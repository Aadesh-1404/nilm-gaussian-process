from data_make import dataset_loader
import os
import pandas as pd

train = {1: {
    'start_time': "2011-04-21",
    'end_time': "2011-05-21"
},
    3: {
    'start_time': "2011-04-19",
    'end_time': "2011-05-22"
},
    2: {
        'start_time': "2011-04-21",
        'end_time': "2011-05-21"
},
    5: {
    'start_time': "2011-04-22",
    'end_time': "2011-06-01"
}
}
test = {6: {
    'start_time': "2011-05-25",
    'end_time': "2011-06-13"
}}
appliances = ["Refrigerator"]


x_train, y_train, x_test, y_test, x_test_timestamp, scaler_x, scaler_y = dataset_loader(
    appliances, train, test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
if not os.path.exists("./datasets/" + appliances[0]):
    os.mkdir("./datasets/" + appliances[0])

pd.DataFrame(x_train).to_csv(
    "./datasets/" + appliances[0] + "/x_train.csv.gz", index=None)
pd.DataFrame(x_test).to_csv(
    "./datasets/" + appliances[0] + "/x_test.csv.gz", index=None)

pd.DataFrame(y_train).to_csv(
    "./datasets/" + appliances[0] + "/y_train.csv.gz", index=None
)
pd.DataFrame(y_test).to_csv(
    "./datasets/" + appliances[0] + "/y_test.csv.gz", index=None)
