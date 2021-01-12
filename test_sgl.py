from lib import sgl
import pandas as pd
import matplotlib.pyplot as plt


def plot_csv():
    df = pd.read_csv("Dati/Dati FLOUD/BCC-Probe1.csv")
    df["measuredTime"] = pd.to_datetime(df["measuredTime"])
    print(df.dtypes)
    df.set_index("measuredTime", inplace=True)
    print(df.head())
    df[["count", "avgSpeed"]].plot()
    plt.show()


def main():
    df = sgl.get_traffic_sensor_df("http://www.disit.org/km4city/resource/iot/iotobsf/GardaLake/Gardesana-BCC-Probe1",
                                "140-day", "2021-01-12T00:00:00")
    df.to_csv("Dati/Dati FLOUD/BCC-Probe1.csv")


if __name__ == "__main__":
    plot_csv()
