import os
import numpy as np
import pandas as pd

def q1a_split_csv_with_train_and_test(filepath):

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

def q1b_regression():
    pass

def q1c_regression_with_regularization():
    pass

def q1d_regression_with_bias():
    pass

def q1e_bayesian_regression():
    pass

def q1f_plot_and_compare():
    pass

def main():
    filepath = "./train.csv"

    if not os.path.exists(filepath):
        q1a_split_csv_with_train_and_test(filepath)

main()
