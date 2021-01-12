from lib import sgl
import pandas as pd
import matplotlib.pyplot as plt


def read_csv():
    df = pd.read_csv("Dati/Dati FLOUD/BCC-Probe1.csv")
    df["measuredTime"] = pd.to_datetime(df["measuredTime"])
    df.set_index("measuredTime", inplace=True)
    df.drop(df.index[df["count"] == 0], inplace=True)
    df = df.resample("H").sum()
    df["avgSpeed"] = df["sumSpeed"] / df["count"]
    return df


def average_week_value():
    df = read_csv()
    # gropy by hour and day of week
    mean_df = df.groupby([df.index.dayofweek, df.index.hour]).mean()
    mean_df.index.names = ["dayofweek", "hour"]
    print(mean_df)
    for index, row in df.iterrows():
        df.loc[index, "week_avg_count"] = mean_df.loc[(index.dayofweek, index.hour), "count"]
        df.loc[index, "week_avg_speed"] = mean_df.loc[(index.dayofweek, index.hour), "sumSpeed"] / mean_df.loc[
            (index.dayofweek, index.hour), "count"]
    df[["count", "week_avg_count"]][df.index >= "2020-12-28"].plot()
    df[["avgSpeed", "week_avg_speed"]][df.index >= "2020-12-28"].plot()
    plt.show()


def main():
    df = sgl.get_traffic_sensor_df("http://www.disit.org/km4city/resource/iot/iotobsf/GardaLake/Gardesana-BCC-Probe1",
                                   "140-day", "2021-01-12T00:00:00")
    df.to_csv("Dati/Dati FLOUD/BCC-Probe1.csv")


if __name__ == "__main__":
    average_week_value()
