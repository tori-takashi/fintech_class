import os
import math
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

    # standardize except for the ID and G3 columns
    ID_column = bin_transformed.ID
    without_ID = bin_transformed.loc[:, bin_transformed.columns != "ID"]

    standardized_columns = pd.DataFrame(scipy.stats.zscore(without_ID),
                                        index=without_ID.index,
                                        columns=without_ID.columns
                                        )
    standardized_with_ID = pd.concat(
<<<<<<< HEAD
        [ID_column, standardized_columns], axis=1)
=======
        [discrete_df, standardized_columns], axis=1)
>>>>>>> 84309e3... add readme, add splitted csv, fix q1a

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


def q1b_regression():
    print("~~~~~q1b~~~~~\n")
    train_data = load_train_data()
    test_data = load_test_data()
    x_train, y_train = split_x_and_y(train_data)
    x_test, y_test = split_x_and_y(test_data)

    # calc w with pseudo-inverse
    w_train = np.linalg.pinv(x_train.T.dot(
        x_train)).dot(x_train.T).dot(y_train)

    show_coefficients(w_train, x_train)
    calc_rmse(w_train, x_test, y_test)


def q1c_regression_with_regularization():
    print("~~~~~q1c~~~~~\n")
    train_data = load_train_data()
    test_data = load_test_data()
    x_train, y_train = split_x_and_y(train_data)
    x_test, y_test = split_x_and_y(test_data)

    lambda_value = 1.0
    w_train = calc_w_train_with_ridge(x_train, y_train, lambda_value)

    show_coefficients(w_train, x_train)
    calc_rmse(w_train, x_test, y_test,
              regularization="ridge", lambda_value=lambda_value)


def q1d_regression_with_bias():
    print("~~~~~q1d~~~~~\n")
    train_data = load_train_data()
    test_data = load_test_data()
    x_train, y_train = split_x_and_y(train_data)
    x_test, y_test = split_x_and_y(test_data)

    bias_train = pd.DataFrame(data=[1]*x_train.shape[0], columns=["bias"])
    x_train_with_bias = pd.concat([bias_train, x_train], axis=1)
    bias_test = pd.DataFrame(data=[1]*x_test.shape[0], columns=["bias"])
    x_test_with_bias = pd.concat([bias_test, x_test], axis=1)

    lambda_value = 1.0
    w_train = calc_w_train_with_ridge(x_train_with_bias, y_train, lambda_value)

    show_coefficients(w_train, x_train_with_bias)
    calc_rmse(w_train, x_test_with_bias, y_test,
              regularization="ridge", lambda_value=lambda_value)


def q1e_bayesian_regression():
    pass


def q1f_plot_and_compare():
    pass


def load_train_data():
    return pd.read_csv("splitted_train_data.csv", index_col=0)


def load_test_data():
    return pd.read_csv("splitted_test_data.csv", index_col=0)


def split_x_and_y(data):
    # split columns
    # remove ID and G3 from dataframe
    x_with_G3 = data.loc[:, data.columns != "ID"]

    x = x_with_G3.loc[:, x_with_G3.columns != "G3"]
    y = data.G3

    return [x, y]


def show_coefficients(w, x):
    print("========coefficients=========")
    x_list = list(x.columns)
    w_list = list(w)

    for i in range(0, len(x_list)-1):
        print(x_list[i] + " : " + str(w_list[i]))


def calc_rmse(w_train, x_test, y_test, regularization="", lambda_value=1):
    if regularization == "ridge":
        y_test_predicted = x_test.dot(w_train)\
            + (lambda_value / 2)*w_train.T.dot(w_train)
    else:
        y_test_predicted = x_test.dot(w_train)

    se_matrix = (y_test - y_test_predicted)**2
    rmse = math.sqrt((1/x_test.shape[0])*sum(se_matrix))
    print("RMSE : " + str(rmse) + "\n")

    return rmse


def calc_w_train_with_ridge(x_train, y_train, lambda_value):
    w_train = np.linalg.inv(x_train.T.dot(x_train)
                            + lambda_value * np.identity(len(x_train.columns)))\
        .dot(x_train.T).dot(y_train)
    return w_train


def main():
    filepath = "./train.csv"
    train_data_path = "./splitted_train_data.csv"
    test_data_path = "./splitted_test_data.csv"

    # solutions
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        q1a_split_csv_with_train_and_test(filepath)
    q1b_regression()
    q1c_regression_with_regularization()
    q1d_regression_with_bias()


main()
