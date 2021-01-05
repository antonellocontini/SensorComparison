import weather_arpav
import matplotlib.pyplot as plt
import pandas as pd


def main():
    f, ax = plt.subplots()
    rain_df = weather_arpav.read_rain_ARPAV_station("ARPAV_BARDOLINO/precipitazioni.csv")
    wind_df = weather_arpav.read_wind_ARPAV_station("ARPAV_BARDOLINO/vel_vento.csv")
    merge_df = weather_arpav.compute_pollutant_dispersion(rain_df, wind_df)
    print(merge_df.head())
    merge_df.plot(ax=ax)
    for index, row in merge_df.iterrows():
        print(index, index + pd.Timedelta(hours=24))
        if row["Dispersion"] == "LOW":
            ax.axvspan(index, index, color="lightcoral")
        elif row["Dispersion"] == "MEDIUM":
            ax.axvspan(index, index, color="lightyellow")
        elif row["Dispersion"] == "HIGH":
            ax.axvspan(index, index, color="lightgreen")
    plt.show()


if __name__ == "__main__":
    main()
