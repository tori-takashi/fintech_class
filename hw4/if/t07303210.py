# (1)
import talib
import pandas as pd
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def normalize(df):
    df_ = df.copy()
    for column in df_.columns:
        min_ = min(df_[column])
        max_ = max(df_[column])
        df_[column] = (df_[column] - min_) / (max_ - min_)
    return df_


def split_train_validation(spy_train, spy_validation):
    x_train = []
    y_train = []
    x_validation = []
    y_validation = []

    for i in range(30, spy_train.values.shape[0]):
        x_train.append(spy_train.values[i-30:i])
        y_train.append(spy_train.values[i, 0])
    for i in range(30, spy_validation.values.shape[0]):
        x_validation.append(spy_validation.values[i-30:i])
        y_validation.append(spy_validation.values[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_validation, y_validation = np.array(x_validation), np.array(y_validation)

    return {"x": x_train, "y": y_train}, {"x": x_validation, "y": y_validation}


spy_ohlcv = pd.read_csv("SPY.csv", date_parser=True)

spy_ohlcv["ma_10"] = talib.MA(
    spy_ohlcv["Close"], timeperiod=10, matype=0).dropna(axis=0)
spy_ohlcv["ma_30"] = talib.MA(
    spy_ohlcv["Close"], timeperiod=30, matype=0).dropna(axis=0)
spy_ohlcv["k_line"], spy_ohlcv["d_line"] = talib.STOCH(
    spy_ohlcv["High"], spy_ohlcv["Low"], spy_ohlcv["Close"],
    fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
spy_ohlcv = spy_ohlcv.dropna(axis=0)

spy_ohlcv_2018 = spy_ohlcv[(spy_ohlcv["Date"] >= "2018-01-01") & (
    spy_ohlcv["Date"] <= "2018-12-31")]

df_2018 = spy_ohlcv_2018.copy()
df_2018["Date"] = df_2018['Date'].astype('datetime64[ns]')
df_2018["Date"] = df_2018["Date"].map(mdates.date2num)

# Moving average 10 and 30
figure_1, ax = plt.subplots()
candlestick_ohlc(ax, df_2018.values,
                 width=5, colorup='g', colordown='r')
ax.xaxis_date()
ax.plot(df_2018["Date"], df_2018["ma_10"], label="MA10")
ax.plot(df_2018["Date"], df_2018["ma_30"], label="MA30")
ax.legend()
plt.show()

# KD Line
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.plot(df_2018["Date"], df_2018["k_line"], label="k_line")
plt.plot(df_2018["Date"], df_2018["d_line"], label="d_line")
plt.legend()
plt.show()

# volume bar
dates = np.asarray(df_2018["Date"])
volume = np.asarray(df_2018["Volume"])

positive = df_2018['Open']-df_2018['Close'] >= 0
negative = df_2018['Open']-df_2018['Close'] < 0

plt.bar(dates[positive], volume[positive], color='green', width=0.7)
plt.bar(dates[negative], volume[negative], color='red', width=0.7)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.show()

# (2) normalize and split
# train
spy_train = spy_ohlcv[spy_ohlcv["Date"] < '2017-01-01'].copy()
spy_train = spy_train.drop(["Date"], axis=1)
spy_train = normalize(spy_train)

# validation
spy_validation = spy_ohlcv[spy_ohlcv["Date"] >= "2017-01-01"].copy()

spy_validation["Date"] = spy_validation["Date"].astype('datetime64[ns]')
spy_validation["Date"] = spy_validation["Date"].map(mdates.date2num)
validation_term = spy_validation["Date"].copy()

spy_validation = spy_validation.drop(["Date"], axis=1)
spy_validation = normalize(spy_validation)

train, validation = split_train_validation(spy_train, spy_validation)

#(3) 
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=8)
def plotModelLoss(history):
    plt.figure(figsize=[9.6,7.2])
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


def plotPrediction(validation_term, validation, model,name="Prediction by RNN"):
    y_pred = model.predict(validation["x"])
    plt.plot(validation_term[30:], validation["y"])
    plt.plot(validation_term[30:], y_pred)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
    plt.gcf().autofmt_xdate()
    plt.title(name)
    plt.legend(["real", "predict"], loc="upper left")
    plt.show()

regressor_RNN = Sequential()
regressor_RNN.add(SimpleRNN(units = 32, activation = 'tanh', input_shape = (train["x"].shape[1], train["x"].shape[2])))
regressor_RNN.add(Dense(units=1))
regressor_RNN.summary()

# simple RNN
checkpoint_RNN = ModelCheckpoint(filepath="best_params_RNN.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
regressor_RNN.compile(optimizer='adam', loss='mean_squared_error')
RNN_history = regressor_RNN.fit(train["x"], train["y"], epochs=256, batch_size=64,
validation_data = (validation["x"], validation["y"]),callbacks=[checkpoint_RNN, early_stopping])

plotModelLoss(RNN_history.history)
plotPrediction(validation_term, validation, regressor_RNN)

# LSTM
regressor_LSTM = Sequential()
regressor_LSTM.add(LSTM(units = 32, activation = 'tanh', input_shape = (train["x"].shape[1], train["x"].shape[2])))
regressor_LSTM.add(Dense(units=1))
regressor_LSTM.summary()

checkpoint_LSTM = ModelCheckpoint(filepath="best_params_LSTM.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
regressor_LSTM.compile(optimizer='adam', loss='mean_squared_error')
LSTM_history = regressor_LSTM.fit(train["x"], train["y"], epochs=256, batch_size=64,
validation_data = (validation["x"], validation["y"]),callbacks=[checkpoint_LSTM, early_stopping])

plotModelLoss(LSTM_history.history)
plotPrediction(validation_term, validation, regressor_LSTM, name="LSTM")

# GRU
regressor_GRU = Sequential()
regressor_GRU.add(GRU(units = 32, activation = 'tanh', input_shape = (train["x"].shape[1], train["x"].shape[2])))
regressor_GRU.add(Dense(units=1))
regressor_GRU.summary()

checkpoint_GRU = ModelCheckpoint(filepath="best_params_GRU.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
regressor_GRU.compile(optimizer='adam', loss='mean_squared_error')
GRU_history = regressor_GRU.fit(train["x"], train["y"], epochs=256, batch_size=64,
validation_data=(validation["x"], validation["y"]), callbacks=[checkpoint_GRU, early_stopping])

plotModelLoss(GRU_history.history)
plotPrediction(validation_term, validation, regressor_GRU, name="GRU")