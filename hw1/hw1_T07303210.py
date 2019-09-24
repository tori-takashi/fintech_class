import numpy as np
import pandas as pd


def split_csv_with_train_and_test(filepath):

    # split randomly and export csv for training and evaluating
    imported_data = pd.read_csv(filepath)

    train_data = imported_data.sample(frac=0.8).sort_values(
        "ID").reset_index().drop("index", axis=1)
    # get 80% data as the training data from "train.csv" with sorted and purified

    train_data_ID = list(train_data["ID"])
    # picking up test data from "train.csv" except for train data
    test_data = imported_data[~imported_data.ID.isin(
        train_data_ID)].sort_values("ID").reset_index().drop("index", axis=1)
    # get 20% data as the test data from "train.csv" with sorted and purified

    train_data.to_csv("splitted_train_data.csv")
    test_data.to_csv("splitted_test_data.csv")
