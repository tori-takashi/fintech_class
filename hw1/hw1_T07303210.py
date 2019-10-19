import os
import math
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt


class Homework1:
    filepath = "train.csv"
    train_data_path = "train_data.csv"
    test_data_path = "test_data.csv"
    standardized_train_path = "standardized_train_data.csv"
    standardized_test_path = "standardized_test_data.csv"
    test_no_G3_path = "test_no_G3.csv"

    y_train_mean = 0.0
    y_train_standard_deviation = 0.0
    y_test_mean = 0.0
    y_test_standard_deviation = 0.0

    y_test_no_G3_mean = 0.0
    y_test_no_G3_standard_deviation = 0.0

    def q1a_split_csv_with_train_and_test(self):
        # split randomly and export csv for training and evaluating

        if not os.path.exists(self.train_data_path) or not os.path.exists(self.test_data_path)\
                or not os.path.exists(self.standardized_train_path) or not os.path.exists(self.standardized_test_path):
            imported_data = pd.read_csv(self.filepath)

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

            # divide train and test data

            train_data_with_ID = bin_transformed.sample(frac=0.8).sort_values(
                "ID").reset_index().drop("index", axis=1)
            test_data_with_ID = bin_transformed.merge(train_data_with_ID, indicator=True, how='outer').query(
                '_merge=="left_only"').drop('_merge', 1).reset_index().drop("index", axis=1)

            train_data = train_data_with_ID[train_data_with_ID.columns[train_data_with_ID.columns != "ID"]]
            test_data = test_data_with_ID[test_data_with_ID.columns[test_data_with_ID.columns != "ID"]]

            standardized_train_data = self.standardize_dataframe(train_data)
            standardized_test_data = self.standardize_dataframe(test_data)

            train_data.to_csv(self.train_data_path)
            test_data.to_csv(self.test_data_path)
            standardized_train_data.to_csv(self.standardized_train_path)
            standardized_test_data.to_csv(self.standardized_test_path)

            self.y_train_mean = np.mean(train_data["G3"])
            self.y_test_mean = np.mean(test_data["G3"])
            self.y_train_standard_deviation = np.std(train_data["G3"])
            self.y_test_standard_deviation = np.std(test_data["G3"])
        else:
            train_data = pd.read_csv(self.train_data_path)
            test_data = pd.read_csv(self.test_data_path)

            self.y_train_mean = np.mean(train_data["G3"])
            self.y_test_mean = np.mean(test_data["G3"])
            self.y_train_standard_deviation = np.std(train_data["G3"])
            self.y_test_standard_deviation = np.std(test_data["G3"])

    def standardize_dataframe(self, df, test_mode=False):
        if test_mode == False:
            G3_column = df.G3
            without_G3 = df.loc[:, df.columns != "G3"]
        else:
            without_G3 = df
        standardized_columns = pd.DataFrame(scipy.stats.zscore(without_G3),
                                            index=without_G3.index,
                                            columns=without_G3.columns
                                            )
        if test_mode == False:
            standardized_with_G3 = pd.concat(
                [G3_column, standardized_columns], axis=1)
            return standardized_with_G3
        else:
            return standardized_columns

    def q1b_regression(self):
        print("~~~~~q1b~~~~~\n")
        data = self.initialize_data(bias=False)

        # calc w with pseudo-inverse
        w_train = np.linalg.pinv(data["x_train"].T.dot(
            data["x_train"])).dot(data["x_train"].T).dot(data["y_train"])

        # self.show_coefficients(w_train, data["x_train"])
        self.calc_rmse(w_train, data["x_test"], data["y_test"])
        return w_train

    def q1c_regression_with_regularization(self):
        print("~~~~~q1c~~~~~\n")
        data = self.initialize_data(bias=False)

        lambda_value = 0.5
        w_train = self.calc_w_train_with_ridge(
            data["x_train"], data["y_train"], lambda_value)

        # self.show_coefficients(w_train, data["x_train"])
        self.calc_rmse(w_train, data["x_test"], data["y_test"])

        return w_train

    def q1d_regression_with_bias(self):
        print("~~~~~q1d~~~~~\n")
        data = self.initialize_data(bias=True)

        lambda_value = 0.5
        w_train = self.calc_w_train_with_ridge(
            data["x_train"], data["y_train"], lambda_value, bias=True)

        # self.show_coefficients(w_train, data["x_train"])
        self.calc_rmse(w_train, data["x_test"], data["y_test"])

        return w_train

    def q1e_bayesian_regression(self):
        print("~~~~~q1e~~~~~\n")
        data = self.initialize_data(bias=True)

        lambda_value = 1.0
        w_train = self.calc_w_train_with_ridge(
            data["x_train"], data["y_train"], lambda_value, bias=True)

        # self.show_coefficients(w_train, data["x_train"])
        self.calc_rmse(w_train, data["x_test"], data["y_test"])

        return w_train

    def q1f_plot_and_compare(self, w_trains):
        data_no_bias = self.initialize_data(bias=False)
        data_with_bias = self.initialize_data(bias=True)

        # q1b => no bias
        # q1c => with bias
        # q1d => with bias
        # q1e => with bias

        x_test = data_no_bias["x_test"]
        x_test_with_bias = data_with_bias["x_test"]

        # y doesn't depend on the bias term
        y_test = data_no_bias["y_test"]

        index = np.arange(0, x_test.shape[0], 1)

        y_predicted_q1b = x_test.dot(w_trains["w_q1b"])
        y_predicted_q1c = x_test.dot(w_trains["w_q1c"])
        y_predicted_q1d = x_test_with_bias.dot(w_trains["w_q1d"])
        y_predicted_q1e = x_test_with_bias.dot(w_trains["w_q1e"])

        rmse_q1b = self.calc_rmse(w_trains["w_q1b"], x_test, y_test)
        rmse_q1c = self.calc_rmse(w_trains["w_q1c"], x_test, y_test)
        rmse_q1d = self.calc_rmse(w_trains["w_q1d"], x_test_with_bias, y_test)
        rmse_q1e = self.calc_rmse(w_trains["w_q1e"], x_test_with_bias, y_test)

        plt.plot(index, y_test, label="Ground Truth")
        plt.plot(index, y_predicted_q1b,
                 label="({}) Linear Regression".format(round(rmse_q1b, 5)))
        plt.plot(index, y_predicted_q1c,
                 label="({})Linear Regression with Reg".format(round(rmse_q1c, 5)))
        plt.plot(index, y_predicted_q1d,
                 label="({}) Linear Regression with Reg and Bias".format(round(rmse_q1d, 5)))
        plt.plot(index, y_predicted_q1e,
                 label="({}) Bayesian Linear Regression".format(round(rmse_q1e, 5)))

        plt.legend(loc='lower center',
                   borderaxespad=0.,
                   ncol=1,
                   mode="expand",
                   title="prediction type",
                   prop={'size': 8})

        plt.title("predicted values")
        plt.xlabel("index")
        plt.ylabel("values")

        plt.savefig("t07303210_figure.png")

        plt.show()

    def q2a_test_no_g3(self, w_q1d):
        print("~~~~~q2a~~~~~\n")
        test_no_g3 = pd.read_csv(self.test_no_G3_path)
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
            "absences"      #
            #"G3"            #
        ]

        designated = test_no_g3.loc[:, column_list]
        bin_transformed = pd.get_dummies(designated, drop_first=True)

        target_data = bin_transformed[bin_transformed.columns[bin_transformed.columns != "ID"]]
        bias = pd.DataFrame(
            data=[1]*target_data.shape[0], columns=["bias"])

        standardized_target_data = self.standardize_dataframe(
            target_data, test_mode=True)
        standardized_target_data_with_bias = pd.concat(
            [bias, standardized_target_data], axis=1)

        predicted_data = standardized_target_data_with_bias.dot(w_q1d)
        predicted_data.to_csv("T07303210_1.txt")

    def load_train_data(self):
        return pd.read_csv(self.standardized_train_path, index_col=0)

    def load_test_data(self):
        return pd.read_csv(self.standardized_test_path, index_col=0)

    def initialize_data(self, bias=False):
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        x_train, y_train = self.split_x_and_y(train_data)
        x_test, y_test = self.split_x_and_y(test_data)

        if bias == True:
            bias_train = pd.DataFrame(
                data=[1]*x_train.shape[0], columns=["bias"])
            x_train_with_bias = pd.concat([bias_train, x_train], axis=1)

            bias_test = pd.DataFrame(
                data=[1]*x_test.shape[0], columns=["bias"])
            x_test_with_bias = pd.concat([bias_test, x_test], axis=1)

            data = {"x_train":  x_train_with_bias, "y_train": y_train,
                    "x_test": x_test_with_bias, "y_test": y_test}
        else:
            data = {"x_train":  x_train, "y_train": y_train,
                    "x_test": x_test, "y_test": y_test}

        return data

    def split_x_and_y(self, data):
        # split columns
        # remove ID and G3 from dataframe
        x_with_G3 = data.loc[:, data.columns != "ID"]

        x = x_with_G3.loc[:, x_with_G3.columns != "G3"]
        y = data.G3

        return [x, y]

    def show_coefficients(self, w, x):
        print("========coefficients=========")
        x_list = list(x.columns)
        w_list = list(w)

        for i in range(0, len(x_list)-1):
            print(x_list[i] + " : " + str(w_list[i]))

    def calc_rmse(self, w_train, x_test, y_test):
        y_test_predicted = x_test.dot(w_train)

        se_matrix = (y_test - y_test_predicted)**2
        rmse = math.sqrt((1/x_test.shape[0])*sum(se_matrix))
        print("RMSE : " + str(rmse) + "\n")

        return rmse

    def calc_w_train_with_ridge(self, x_train, y_train, lambda_value, bias=False):
        if bias == True:
            identity = np.identity(len(x_train.columns))
            identity[0][0] = 0.0
            identity_for_ridge = identity
        else:
            identity_for_ridge = np.identity(len(x_train.columns))

        w_train = np.linalg.inv(x_train.T.dot(x_train)
                                + lambda_value * identity_for_ridge
                                ).dot(x_train.T).dot(y_train)
        return w_train

    def main(self):

        # solutions
        self.q1a_split_csv_with_train_and_test()

        w_q1b = self.q1b_regression()
        w_q1c = self.q1c_regression_with_regularization()
        w_q1d = self.q1d_regression_with_bias()
        w_q1e = self.q1e_bayesian_regression()
        w_trains = {"w_q1b": w_q1b, "w_q1c": w_q1c,
                    "w_q1d": w_q1d, "w_q1e": w_q1e}
        self.q1f_plot_and_compare(w_trains)

        self.q2a_test_no_g3(w_q1d)


homework1 = Homework1()
homework1.main()
