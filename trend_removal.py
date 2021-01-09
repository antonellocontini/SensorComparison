from lib import ibe, analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


def linear_regression_parameters(ser, start_time):
    regr = linear_model.LinearRegression()
    X = ((ser.dropna().index - start_time) / pd.Timedelta(hours=1)).to_numpy().reshape(-1, 1)
    y = ser.dropna().values
    # print(X)
    # print(y)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    # plt.scatter(X, y)
    # plt.plot(X, y_pred)
    df = ser.to_frame()
    df["coef"] = regr.coef_[0]
    df["intercept"] = regr.intercept_
    return df


# use segmented regression
def find_trend_model(ibe_ser, freq="M"):
    segments = [g for n, g in ibe_ser.groupby(pd.Grouper(freq=freq))]
    start_time = segments[0].index[0]
    result_df = None
    for s in segments:
        trended = s
        df = linear_regression_parameters(trended, start_time)
        if result_df is None:
            result_df = df.copy(deep=True)
        else:
            result_df = pd.concat([result_df, df])
    result_df["trend"] = ((result_df.index - start_time) / pd.Timedelta(hours=1)) * result_df["coef"] + result_df["intercept"]
    return result_df
    f, ax = plt.subplots()
    result_df["O3"].plot(ax=ax)
    detrended = result_df["O3"] - result_df["trend"]
    detrended.plot(ax=ax)
    result_df["trend"].plot(ax=ax)
    # valuta rimozione modello con Dickey-Fuller
    adf_original = adfuller(ibe_ser.dropna().values)
    print("original:", adf_original[0], adf_original[1])
    adf_detrended = adfuller(detrended.dropna().values)
    print("detrended:", adf_detrended[0], adf_detrended[1])
    # plt.show()


def main():
    ibe_df = ibe.read_IBE_sensor("Dati/Dati IBE/SMART53.json", "Dati/Dati IBE/SMART53.params.json")
    ibe_df = analysis.outlier_removal(ibe.clip_IBE_data(ibe_df))
    # month_df = find_trend_model(ibe_df["O3"], freq="M")
    week_df = find_trend_model(ibe_df["O3"], freq="W")
    detrended_ser = week_df["O3"] - week_df["trend"]
    detrended_ser.plot()
    ac = pd.DataFrame([detrended_ser.autocorr(i) for i in range(1, 25)])
    ac.index += 1
    pd.DataFrame(ac).plot(kind="bar")
    deseasonalized_ser = detrended_ser - detrended_ser.shift(24)
    adf_original = adfuller(week_df["O3"].dropna().values)
    print("ADF original:", adf_original[0], "p-value:", adf_original[1])
    adf_detrended = adfuller(detrended_ser.dropna().values)
    print("ADF detrended:", adf_detrended[0], "p-value:", adf_detrended[1])
    adf_deseasonalized = adfuller(deseasonalized_ser.dropna().values)
    print("ADF deseasonalied:", adf_deseasonalized[0], "p-value:", adf_deseasonalized[1])
    f, ax = plt.subplots(nrows=3)
    week_df["O3"].plot(ax=ax[0])
    detrended_ser.plot(ax=ax[1])
    deseasonalized_ser.plot(ax=ax[2])
    plt.show()


if __name__ == "__main__":
    main()
