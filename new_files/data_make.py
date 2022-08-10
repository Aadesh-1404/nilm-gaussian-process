import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch

# device = "cuda"


# def dataset_loader(appliances, train, test=None):
#     x_train = []
#     y_train = []
#     scaler_x = StandardScaler()
#     scaler_y = StandardScaler()
#     # train
#     for key, values in train.items():
#         df = pd.read_csv(
#             f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp", "main", appliances[0]])
#         df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
#         startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
#         endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()

#         if startDate > endDate:
#             raise "Start Date must be smaller than Enddate."

#         df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
#         df.dropna(inplace=True)
#         x = df["main"].values
#         y = df[appliances[0]].values

#         x_train.extend(np.array(x))
#         y_train.extend(np.array(y))

#     x_train = np.array(x_train).reshape(-1, 1)
#     y_train = np.array(y_train).reshape(-1, 1)
#     x_train = scaler_x.fit_transform(x_train)
#     y_train = scaler_y.fit_transform(y_train)

#     # test
#     x_test = []
#     y_test = []
#     x_test_timestamp = []
#     for key, values in test.items():
#         df = pd.read_csv(
#             f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp", "main", appliances[0]])
#         df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
#         startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
#         endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()

#         if startDate > endDate:
#             raise "Start Date must be smaller than Enddate."

#         df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
#         df.dropna(inplace=True)
#         x = df["main"].values
#         y = df[appliances[0]].values
#         timestamp = df["Timestamp"].values

#         x_test.extend(np.array(x))
#         y_test.extend(np.array(y))
#         x_test_timestamp.extend(timestamp)

#     x_test = np.array(x_test).reshape(-1, 1)
#     y_test = np.array(y_test).reshape(-1, 1)

#     x_test = scaler_x.transform(x_test)

#     x_train = np.array(x_train).reshape(x_train.shape[0], 1)
#     y_train = np.array(y_train).reshape(-1,)
#     x_test = np.array(x_test).reshape(x_test.shape[0], 1)
#     y_test = np.array(y_test).reshape(-1,)

#     return x_train, y_train, x_test, y_test, x_test_timestamp, scaler_x, scaler_y

def dataset_loader(appliances, train, test=None):
    x_train = []
    y_train = []
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    # train
    for key, values in train.items():
        df = pd.read_csv(
            f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp", "main", appliances[0]])
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
        endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()

        if startDate > endDate:
            raise "Start Date must be smaller than Enddate."

        df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
        df.dropna(inplace=True)
        x = df["main"].values
        y = df[appliances[0]].values

        x_train.extend(torch.tensor(x))
        y_train.extend(torch.tensor(y))

    x_train = torch.tensor(x_train).reshape(-1, 1)
    y_train = torch.tensor(y_train).reshape(-1, 1)
    x_train = scaler_x.fit_transform(x_train)
    y_train = scaler_y.fit_transform(y_train)

    # test
    x_test = []
    y_test = []
    x_test_timestamp = []
    for key, values in test.items():
        df = pd.read_csv(
            f"datasets/Building{key}_NILM_data_basic.csv", usecols=["Timestamp", "main", appliances[0]])
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        startDate = datetime.strptime(values["start_time"], "%Y-%m-%d").date()
        endDate = datetime.strptime(values["end_time"], "%Y-%m-%d").date()

        if startDate > endDate:
            raise "Start Date must be smaller than Enddate."

        df = df[(df["date"] >= startDate) & (df["date"] <= endDate)]
        df.dropna(inplace=True)
        x = df["main"].values
        y = df[appliances[0]].values
        timestamp = df["Timestamp"].values

        x_test.extend(torch.tensor(x))
        y_test.extend(torch.tensor(y))
        x_test_timestamp.extend(timestamp)

    x_test = torch.tensor(x_test).reshape(-1, 1)
    y_test = torch.tensor(y_test).reshape(-1, 1)

    x_test = scaler_x.transform(x_test)

    x_train = torch.tensor(x_train).reshape(x_train.shape[0], 1)
    y_train = torch.tensor(y_train).reshape(-1,)
    x_test = torch.tensor(x_test).reshape(x_test.shape[0], 1)
    y_test = torch.tensor(y_test).reshape(-1,)

    return x_train, y_train, x_test, y_test, x_test_timestamp, scaler_x, scaler_y
