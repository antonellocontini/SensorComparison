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


def read_traffic_csv():
    df = pd.read_csv("Dati/Dati FLOUD/BCC-Probe1.csv")
    df["measuredTime"] = pd.to_datetime(df["measuredTime"])
    df.set_index("measuredTime", inplace=True)
    df.drop(df.index[df["count"] == 0], inplace=True)
    # resample on hours
    df = df.resample("H").sum()
    df = analysis.outlier_removal(df)
    df["avgSpeed"] = df["sumSpeed"] / df["count"]
    return df


def autocorr(ser):
    ac = pd.DataFrame([ser.autocorr(i) for i in range(1, 25)])
    ac.index += 1
    pd.DataFrame(ac).plot(kind="bar")


def average_week_value(df):
    # gropy by hour and day of week
    mean_df = df.groupby([df.index.dayofweek, df.index.hour]).mean()
    mean_df.index.names = ["dayofweek", "hour"]
    for index, row in df.iterrows():
        df.loc[index, "week_avg_count"] = mean_df.loc[(index.dayofweek, index.hour), "count"]
        df.loc[index, "week_avg_speed"] = mean_df.loc[(index.dayofweek, index.hour), "sumSpeed"] / mean_df.loc[
            (index.dayofweek, index.hour), "count"]
    return df


def main():
    traffic_df = read_traffic_csv()
    traffic_df = average_week_value(traffic_df)
    traffic_df = traffic_df[(traffic_df.index >= "2020-09-20") & (traffic_df.index <= "2020-10-10")]
    deseasonalized_traffic_ser = traffic_df["count"] - traffic_df["count"].shift(24)
    deseasonalized_traffic_ser2 = traffic_df["count"] - traffic_df["week_avg_count"]
    ibe_df = ibe.read_IBE_sensor("Dati/Dati IBE/SMART53.json", "Dati/Dati IBE/SMART53.params.json")
    ibe_df = analysis.outlier_removal(ibe.clip_IBE_data(ibe_df))
    # month_df = find_trend_model(ibe_df["O3"], freq="M")
    week_df = find_trend_model(ibe_df["O3"], freq="W")
    detrended_ibe_ser = week_df["O3"] - week_df["trend"]
    # autocorr(detrended_ibe_ser)
    # autocorr(traffic_df["count"])
    deseasonalized_ibe_ser = detrended_ibe_ser - detrended_ibe_ser.shift(24)
    adf_original = adfuller(week_df["O3"].dropna().values)
    print("ADF original:", adf_original[0], "p-value:", adf_original[1])
    adf_detrended = adfuller(detrended_ibe_ser.dropna().values)
    print("ADF detrended:", adf_detrended[0], "p-value:", adf_detrended[1])
    adf_deseasonalized = adfuller(deseasonalized_ibe_ser.dropna().values)
    print("ADF deseasonalied:", adf_deseasonalized[0], "p-value:", adf_deseasonalized[1])
    f, ax = plt.subplots(nrows=3)
    week_df["O3"].plot(ax=ax[0])
    detrended_ibe_ser.plot(ax=ax[1])
    deseasonalized_ibe_ser.plot(ax=ax[2])

    f2, ax2 = plt.subplots(nrows=3)
    traffic_df["count"].plot(ax=ax2[0])
    deseasonalized_traffic_ser.plot(ax=ax2[1])
    deseasonalized_traffic_ser2.plot(ax=ax2[2])
    plt.show()


if __name__ == "__main__":
    main()
