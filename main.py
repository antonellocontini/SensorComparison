import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
from sklearn.metrics import mean_squared_error


def convert_IBE_json_to_df(js, calibration_params):
    df = pd.DataFrame.from_dict(js["data"])
    columns = ['y_coord', 'RH', 'PM10', 'CO2', 'PM2.5', 'x_coord', 'O3', 'VOC',
               'NO_A', 'T', 'NO2', 'CO', 'NO2_A']
    for c in columns:
        df[c] = df[c].astype("float64")
    for v in calibration_params:
        params = calibration_params[v]
        df[v] = params["a"] * df[v] + \
            params["b"] * df["T"] + \
            params["c"] * df[v] * df["T"] + \
            params["q"]
    df["data"] = pd.to_datetime(df["data"], utc=True)
    df.set_index("data", inplace=True)
    return df


def read_IBE_sensor(data_filename, params_filename):
    with open(params_filename) as f:
        calibration_params = json.load(f)
    with open(data_filename) as f:
        js = json.load(f)
    df = convert_IBE_json_to_df(js, calibration_params)
    df = df.resample("H").mean()
    # remove outliers
    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    return df


def read_ARPAV_station(data_filename):
    df = pd.read_csv(data_filename)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H", utc=True)
    df.set_index("Datetime", inplace=True)
    return df


def pearson_coefficient_common_variables(arpav_df, ibe_df, window=24):
    def pearson(ser):
        try:
            r, p = stats.pearsonr(arpav_df[v][ser.index].dropna(), ibe_df[v][ser.index].dropna())
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    def rmse(ser):
        try:
            r = mean_squared_error(arpav_df[v][ser.index].dropna(),
                                                                ibe_df[v][ser.index].dropna(), squared=False)
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    # f, ax = plt.subplots(nrows=2, ncols=3)
    variables = [["NO2", "O3", "CO"], ["T", "RH", None]]
    i = 0
    for l in variables:
        j = 0
        for v in l:
            if v is not None:
                f, ax = plt.subplots(nrows=3, sharex=True)
                print(v)
                rol = arpav_df[v].rolling(window=window)
                rol.apply(pearson, raw=False).plot(ax=ax[0])
                ax[0].set_title(f"Moving window pearson R - {v}")
                rol.apply(rmse, raw=False).plot(ax=ax[1])
                ax[1].set_title(f"Moving window RMSE - {v}")
                arpav_df[v].plot(ax=ax[2], label=f"ARPAV {v}")
                ibe_df[v].plot(ax=ax[2], label=f"IBE {v}")
                ax[2].legend()
                plt.tight_layout()
            j = j + 1
        i = i + 1
    plt.show()
        # merge_df = pd.merge(arpav_df[v], ibe_df[v], left_index=True, right_index=True, how="outer")
        # stats.pearsonr(arpav_df[x], ibe_df[x])

    # r, p = stats.pearsonr(df.dropna()["count"], df.dropna()["PM10"])
    # print(f"[SCIPY] # of vehicles over PM10 - Pearson r: {r} and p-value: {p}")
    # r, p = stats.pearsonr(df.dropna()["count"], df.dropna()["PM25"])
    # print(f"[SCIPY] # of vehicles over PM2.5 - Pearson r: {r} and p-value: {p}")
    # r, p = stats.pearsonr(df.dropna()["count"], df.dropna()["CO2"])
    # print(f"[SCIPY] # of vehicles over CO2 - Pearson r: {r} and p-value: {p}")
    # r, p = stats.pearsonr(df.dropna()["avgSpeed"], df.dropna()["PM10"])
    # print(f"[SCIPY] average speed over PM10 - Pearson r: {r} and p-value: {p}")
    # r, p = stats.pearsonr(df.dropna()["avgSpeed"], df.dropna()["PM25"])
    # print(f"[SCIPY] average speed over PM2.5 - Pearson r: {r} and p-value: {p}")
    # r, p = stats.pearsonr(df.dropna()["avgSpeed"], df.dropna()["CO2"])
    # print(f"[SCIPY] average speed over CO2 - Pearson r: {r} and p-value: {p}")
    # return panda_r


def plot_common_variables(arpav_df, ibe_df, show=False):
    # f, ax = plt.subplots(2, 3)
    variables = [["NO2", "O3", "CO"], ["T", "RH", None]]
    i = 0
    for l in variables:
        j = 0
        for v in l:
            if v is not None:
                f2, ax2 = plt.subplots()
                merge_df = pd.merge(arpav_df[v], ibe_df[v], left_index=True, right_index=True, how="outer")
                merge_df.plot(ax=ax2)
                plt.tight_layout()
            j = j + 1
        i = i + 1
    if show:
        plt.show()


def main():
    sensor_name = "SMART53"
    ibe_df = read_IBE_sensor(f"{sensor_name}.json", f"{sensor_name}.params.json")
    restricted_ibe_df = ibe_df[(ibe_df.index > f"2020-07-01") & (ibe_df.index < f"2020-07-27")]

    arpav_df = read_ARPAV_station("MMC.csv")
    restricted_arpav_df = arpav_df[(arpav_df.index > f"2020-07-01") & (arpav_df.index < f"2020-07-27")]
    # plot_common_variables(restricted_arpav_df, restricted_ibe_df, True)
    pearson_coefficient_common_variables(restricted_arpav_df, restricted_ibe_df)


if __name__ == '__main__':
    main()
