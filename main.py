import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import stats
from sklearn.metrics import mean_squared_error
from pathlib import Path


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
    # df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H", utc=True)

    # leggo timestamp in UTC+2 (CEST) e converto in UTC
    # N.B. faccio questo perchè c'è un bug nel plot delle timeseries dove la timezone
    # non viene considerata
    tz_naive = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H")
    df["Datetime"] = tz_naive.dt.tz_localize(tz="Europe/Rome").dt.tz_convert(tz="UTC")
    df.set_index("Datetime", inplace=True)
    return df


# Confronto arpav con ibe
# Si passano due dataframe e l'elenco di colonne in comune
# i due dataframe devono avere la stessa frequenza temporale
# vedi df.resample()
def similarity_common_variables(arpav_df, ibe_df, ibe_name, variables=None, window=24, save_graphs=True, show=True):
    def pearson(ser):
        try:
            arpav_ser = arpav_df[v][ser.index].dropna()
            ibe_ser = ibe_df[v][ser.index].dropna()
            if arpav_ser.nunique() <= 1 or ibe_ser.nunique() <= 1:
                return np.nan
            r, p = stats.pearsonr(arpav_ser, ibe_ser)
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    def rmse(ser):
        try:
            # RMSE (squared=False)
            # sklearn calcola MSE = sum_i^n((arpav[i] - ibe[i])**2)/n
            r = mean_squared_error(arpav_df[v][ser.index].dropna(),
                                                                ibe_df[v][ser.index].dropna(), squared=False)
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    def nrmse(ser):
        min_value = arpav_df[v].min()
        max_value = arpav_df[v].max()
        try:
            # RMSE (squared=False)
            r = mean_squared_error(arpav_df[v][ser.index].dropna(),
                                                                ibe_df[v][ser.index].dropna(), squared=False)
            nr = r * 100 / (max_value - min_value)
            return nr
        except (KeyError, ValueError) as e:
            return np.nan

    units = {
        "NO2": "µg/m3",
        "O3": "µg/m3",
        "CO": "mg/m3",
        "T": "C°",
        "RH": "%"
    }

    if variables is None:
        variables = ["NO2", "O3", "CO", "T", "RH"]

    pearson_rs = {}
    for v in variables:
        if v in arpav_df and v in ibe_df:
            unique_df = pd.concat([arpav_df[v].rename(f"ARPAV {v}"), ibe_df[v].rename(f"IBE {v}")], axis=1).dropna()
            f, ax = plt.subplots(nrows=3, sharex=True)
            pearson_ax = ax[0]
            rmse_ax = ax[1]
            data_ax = ax[2]
            print(v)
            rol = arpav_df[v].rolling(window=window)

            rol.apply(pearson, raw=False).plot(ax=pearson_ax)
            overall_r, overall_p = stats.pearsonr(unique_df[f"ARPAV {v}"], unique_df[f"IBE {v}"])
            overall_r = overall_r
            pearson_rs[v] = overall_r
            pearson_ax.set_ylim([-1, 1])
            pearson_ax.axhline(overall_r, color="red", linestyle="--")
            pearson_ax.set_title(f"{v} - Moving window pearson R - Overall R: {overall_r}")

            rmse_ser = rol.apply(rmse, raw=False)
            # rmse_avg = rmse_ser.mean()
            overall_rmse = mean_squared_error(unique_df[f"ARPAV {v}"], unique_df[f"IBE {v}"], squared=False)
            rmse_ser.plot(ax=rmse_ax)
            rmse_ax.axhline(overall_rmse, color="red", linestyle="--")
            rmse_ax.set_title(f"{v} - Moving window RMSE - Overall RMSE: {overall_rmse:.3f}")
            if v in units:
                rmse_ax.set_ylabel(units[v])

            # nrmse_ser = rol.apply(nrmse, raw=False)
            # overall_nrmse = mean_squared_error(unique_df[f"ARPAV {v}"], unique_df[f"IBE {v}"], squared=False) / (unique_df)
            # nrmse_avg = nrmse_ser.mean()
            # nrmse_ser.plot(ax=ax[2])
            # ax[2].set_title(f"Mov. window Normalized RMSE - {v} Average NRMSE: {nrmse_avg:.3f}%")
            # ax[2].set_ylabel("%")

            arpav_df[v].plot(ax=data_ax, label=f"ARPAV {v}")
            ibe_df[v].plot(ax=data_ax, label=f"IBE {v}")
            data_ax.set_title(f"ARPAV vs IBE {ibe_name}")
            if v in units:
                data_ax.set_ylabel(units[v])
            data_ax.legend()
            plt.tight_layout()
            if save_graphs:
                graph_directory = Path(f"similarity_graphs/{ibe_name}")
                graph_directory.mkdir(parents=True, exist_ok=True)
                graph_filename = graph_directory.joinpath(f"{v}.png")
                plt.savefig(graph_filename)

    if len(pearson_rs) > 0:
        summary_f, summary_ax = plt.subplots()
        pearson_df = pd.DataFrame.from_dict(pearson_rs, orient="index", columns=["Pearson"])
        pearson_df.plot.bar(ax=summary_ax)
        summary_ax.set_ylim([-1, 1])
        summary_ax.set_title(f"Correlation between {ibe_name} and ARPAV")
        for p in summary_ax.patches:
            h = p.get_height()
            if h > 0:
                label_y = -0.12
            else:
                label_y = 0.05
            summary_ax.annotate(f"{p.get_height():.3f}", (p.get_x()+p.get_width()/2., label_y), ha="center")
    if show:
        plt.show()


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
    print(restricted_arpav_df)
    # plot_common_variables(restricted_arpav_df, restricted_ibe_df, True)
    similarity_common_variables(restricted_arpav_df, restricted_ibe_df, sensor_name)


if __name__ == '__main__':
    main()
