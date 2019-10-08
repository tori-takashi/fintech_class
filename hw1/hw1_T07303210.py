import os
import numpy as np
import pandas as pd
import scipy.stats


def q1a_split_csv_with_train_and_test(filepath):

    # split randomly and export csv for training and evaluating
    imported_data = pd.read_csv(filepath)

    # set designated columns
    column_list = [
        "ID",           #
        "school",       # (bin) MS=1  GP=0 => school_MP
        "sex",          # (bin) M=1   F=0  => sex_M
        "age",          #
        "famsize",      # (bin) LE3=1 ME=0 => famsize_LE3
        "studytime",    #
        "failures",     #
        "activities",   # (bin) yes=1 no=0 => activities_yes
        "higher",       # (bin) yes=1 no=0 => higher_yes
        "internet",     # (bin) yes=1 no=0 => internet_yes
        "romantic",     # (bin) yes=1 no=0 => romantic_yes
        "famrel",       #
        "freetime",     #
        "goout",        #
        "Dalc",         #
        "Walc",         #
        "health",       #
        "absences",     #
        "G3"            #
    ]

    # create dataframe with designated columns in the homework
    designated = imported_data.loc[:, column_list]
    bin_transformed = pd.get_dummies(designated, drop_first=True)

    # standardize except for the ID column
    ID_column = bin_transformed.ID
    without_ID = bin_transformed.loc[:, bin_transformed.columns != "ID"]

    standardized = pd.DataFrame(scipy.stats.zscore(without_ID),
                                index=without_ID.index,
                                columns=without_ID.columns
                                )
    standardized_with_ID = pd.concat([ID_column, standardized], axis=1)

    # generate train and test data
    # get 80% data as the training data from "train.csv" with sorted and purified
    train_data = standardized_with_ID.sample(frac=0.8).sort_values(
        "ID").reset_index().drop("index", axis=1)

    # picking up test data from "train.csv" except for train data
    # get 20% data as the test data from "train.csv" with sorted and purified
    train_data_ID = list(train_data["ID"])
    test_data = standardized_with_ID[~standardized_with_ID.ID.isin(
        train_data_ID)].sort_values("ID").reset_index().drop("index", axis=1)

    # create csv files
    train_data.to_csv("splitted_train_data.csv")
    test_data.to_csv("splitted_test_data.csv")


def load_train_data():
    return pd.read_csv("splitted_train_data.csv")


def load_test_data():
    return pd.read_csv("splitted_test_data.csv")


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
    train_data_path = "./splitted_train_data.csv"
    test_data_path = "./splitted_test_data.csv"

    if not os.path.exists(train_data_path) or os.path.exists(test_data_path):
        q1a_split_csv_with_train_and_test(filepath)


main()
