from lib import ibe, analysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller


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
def find_trend_model(ibe_ser, freq="3W"):
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
    f, ax = plt.subplots()
    result_df["O3"].plot(ax=ax)
    detrended = result_df["O3"] - (
                ((result_df.index - start_time) / pd.Timedelta(hours=1)) * result_df["coef"] + result_df["intercept"])
    detrended.plot(ax=ax)
    model = ((result_df.index - start_time) / pd.Timedelta(hours=1)) * result_df["coef"] + result_df["intercept"]
    model.plot(ax=ax)
    # valuta rimozione modello con Dickey-Fuller
    adf_original = adfuller(ibe_ser.dropna().values)
    print("original:", adf_original[0], adf_original[1])
    adf_detrended = adfuller(detrended.dropna().values)
    print("detrended:", adf_detrended[0], adf_detrended[1])
    plt.show()


def main():
    ibe_df = ibe.read_IBE_sensor("Dati/Dati IBE/SMART53.json", "Dati/Dati IBE/SMART53.params.json")
    ibe_df = analysis.outlier_removal(ibe.clip_IBE_data(ibe_df))
    find_trend_model(ibe_df["O3"])


if __name__ == "__main__":
    main()
