import weather_arpav
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn


def plot_as_heatmap(weather_df, ax):
    ser = weather_df["Dispersion"]
    ser = ser.map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    n = len(ser)
    np_array = ser.to_numpy()
    print(n)
    if n % 7 != 0:
        np_array = np.append(np_array, np.zeros(7 - (n % 7)) + np.nan)
        print(len(np_array))
    by_weeks = np_array.reshape((-1, 7))
    seaborn.heatmap(by_weeks, cmap="hot", ax=ax)


def main():
    f, ax = plt.subplots()
    rain_df = weather_arpav.read_rain_ARPAV_station("ARPAV_BARDOLINO/precipitazioni.csv")
    wind_df = weather_arpav.read_wind_ARPAV_station("ARPAV_BARDOLINO/vel_vento.csv")
    weather_df = weather_arpav.compute_pollutant_dispersion(rain_df, wind_df)
    weather_df.to_csv("pollutant_dispersion.csv")
    f2, ax2 = plt.subplots()
    plot_as_heatmap(weather_df, ax=ax2)
    weather_df.plot(ax=ax)
    for index, row in weather_df.iterrows():
        if row["Dispersion"] == "LOW":
            ax.axvspan(index, index, color="lightcoral")
        elif row["Dispersion"] == "MEDIUM":
            ax.axvspan(index, index, color="lightyellow")
        elif row["Dispersion"] == "HIGH":
            ax.axvspan(index, index, color="lightgreen")
    plt.show()


if __name__ == "__main__":
    main()
