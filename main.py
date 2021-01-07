import analysis
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from pathlib import Path

import arpav
import weather_arpav
import ibe


def arpav_ibe_comparison():
    rain_df = weather_arpav.read_rain_ARPAV_station("Dati/ARPAV_BARDOLINO/precipitazioni.csv")
    wind_df = weather_arpav.read_wind_ARPAV_station("Dati/ARPAV_BARDOLINO/vel_vento.csv")
    weather_df = weather_arpav.compute_pollutant_dispersion(rain_df, wind_df)
    folder = "similarity_graphs"
    from_date = "2020-07-01"
    to_date = "2020-07-26"
    restricted_weather_df = weather_df[(weather_df.index >= from_date) & (weather_df.index <= to_date)]
    smart53_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/SMART53.json", f"Dati/Dati IBE/SMART53.params.json")
    restricted_smart53_df = smart53_df[(smart53_df.index > from_date) & (smart53_df.index < to_date)]
    smart54_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/SMART54.json", f"Dati/Dati IBE/SMART54.params.json")
    restricted_smart54_df = smart54_df[(smart54_df.index > from_date) & (smart54_df.index < to_date)]

    # clip IBE data
    restricted_smart53_df = ibe.clip_IBE_data(restricted_smart53_df)
    restricted_smart54_df = ibe.clip_IBE_data(restricted_smart54_df)
    # remove outliers
    restricted_smart53_df = analysis.outlier_removal(restricted_smart53_df)
    restricted_smart54_df = analysis.outlier_removal(restricted_smart54_df)

    arpav_df = arpav.read_ARPAV_station("Dati/Unità mobile - Garda/MMC.csv")
    restricted_arpav_df = arpav_df[(arpav_df.index > from_date) & (arpav_df.index < to_date)]
    arpav_53_similarity = analysis.similarity_common_variables(restricted_arpav_df, restricted_smart53_df, "ARPAV",
                                                               "SMART53", show=False, folder=folder,
                                                               weather_df=restricted_weather_df)[2:]
    arpav_54_similarity = analysis.similarity_common_variables(restricted_arpav_df, restricted_smart54_df, "ARPAV",
                                                               "SMART54", show=False, folder=folder,
                                                               weather_df=restricted_weather_df)[2:]
    ibe_similarity = analysis.similarity_common_variables(restricted_smart53_df, restricted_smart54_df, "SMART53",
                                                          "SMART54", show=False, folder=folder,
                                                          weather_df=restricted_weather_df)[2:]

    graph_directory = Path(folder)
    graph_directory.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    titles = ["Pearson", "RMSE"]
    for i, title in enumerate(titles):
        f, ax = plt.subplots()
        pd.concat([arpav_53_similarity[i].rename(columns={f"{title}": "ARPAV v SMART53"}),
                   arpav_54_similarity[i].rename(columns={f"{title}": "ARPAV v SMART54"}),
                   ibe_similarity[i].rename(columns={f"{title}": "SMART53 v SMART54"})], axis=1).plot.bar(ax=ax)
        ax.set_title(title)
        ax_range = ax.get_ylim()
        ax_range = ax_range[1] - ax_range[0]
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                label_y = h + ax_range * 0.02
            else:
                label_y = h - ax_range * 0.02
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., label_y), ha="center", va="center",
                        fontsize=4)
        ax.legend(prop={'size': 6})
        plt.tight_layout()
        graph_filename = graph_directory.joinpath(f"{title} - ARPAV v IBE.png")
        plt.savefig(graph_filename, dpi=300)
        plt.close("all")


def ibe_monthly_analysis(ibe_a_name, ibe_b_name):
    print(f"Start month analysis between {ibe_a_name} and {ibe_b_name}")
    rain_df = weather_arpav.read_rain_ARPAV_station("Dati/ARPAV_BARDOLINO/precipitazioni.csv")
    wind_df = weather_arpav.read_wind_ARPAV_station("Dati/ARPAV_BARDOLINO/vel_vento.csv")
    weather_df = weather_arpav.compute_pollutant_dispersion(rain_df, wind_df)
    a_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/{ibe_a_name}.json", f"Dati/Dati IBE/{ibe_a_name}.params.json")
    b_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/{ibe_b_name}.json", f"Dati/Dati IBE/{ibe_b_name}.params.json")
    units = {
        "NO2": "µg/m3",
        "O3": "µg/m3",
        "CO": "mg/m3",
        "T": "C°",
        "RH": "%",
        "PM10": "µg/m3",
        "PM2.5": "µg/m3",
        "CO2": "ppm"
    }
    for month in range(7, 13, 2):
        from_date = pd.to_datetime(f"2020-{month:02}-01", utc=True)
        to_date = pd.to_datetime(f"2020-{month:02}-01", utc=True) + MonthEnd(2)
        restricted_weather_df = weather_df[(weather_df.index >= from_date) & (weather_df.index <= to_date)]
        restricted_a_df = a_df[(a_df.index > from_date) & (a_df.index < to_date)]
        restricted_b_df = b_df[(b_df.index > from_date) & (b_df.index < to_date)]
        restricted_a_df = ibe.clip_IBE_data(restricted_a_df)
        restricted_b_df = ibe.clip_IBE_data(restricted_b_df)
        restricted_a_df = analysis.outlier_removal(restricted_a_df)
        restricted_b_df = analysis.outlier_removal(restricted_b_df)
        analysis.similarity_common_variables(restricted_a_df, restricted_b_df, ibe_a_name, ibe_b_name,
                                             show=False,
                                             variables=["NO2", "O3", "CO", "T", "RH", "CO2", "PM10", "PM2.5"],
                                             units=units,
                                             folder=f"IBE comparison - 2020-{month:02}",
                                             weather_df=restricted_weather_df)
        print(f"IBE comparison - 2020-{month:02} completed")
        plt.close("all")


def ibe_trend():
    limits = {
        "O3": 300,
        "NO2": 300,
        "CO": 30,
        "PM2.5": 300,
        "PM10": 300,
        "CO2": 1000
    }
    smart53_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/SMART53.json", f"Dati/Dati IBE/SMART53.params.json")
    smart54_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/SMART54.json", f"Dati/Dati IBE/SMART54.params.json")
    smart53_df = ibe.clip_IBE_data(smart53_df)
    smart53_df = analysis.outlier_removal(smart53_df)
    smart54_df = ibe.clip_IBE_data(smart54_df)
    smart54_df = analysis.outlier_removal(smart54_df)
    freq = pd.offsets.Week(weekday=0, n=2)
    # freq = "MS"
    analysis.bar_plot_resampled(smart53_df, freq, show=False, name="SMART53",
                                variables=["NO2", "O3", "CO", "T", "RH", "PM10", "PM2.5"])
    analysis.bar_plot_resampled(smart53_df, freq, show=True, name="SMART53",
                                variables=["CO2"])


# plotta il grafico delle differenze tra due sensori IBIMET
# è possibile indicare le variabili da plottare come lista di stringhe
# e le loro unità di misura come dizionario
# è possibile anche limitare il periodo di analisi dei dati passando
# a from_date e to_date delle stringhe in formato "AAAA-MM-GG"
def ibe_plot_difference(ibe_a_name, ibe_b_name, variables=None, units=None, from_date=None, to_date=None):
    if units is None:
        units = {
            "NO2": "µg/m3",
            "O3": "µg/m3",
            "CO": "mg/m3",
            "T": "C°",
            "RH": "%",
            "PM10": "µg/m3",
            "PM2.5": "µg/m3",
            "CO2": "ppm"
        }
    ibe_a_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/{ibe_a_name}.json", f"Dati/Dati IBE/{ibe_a_name}.params.json")
    ibe_b_df = ibe.read_IBE_sensor(f"Dati/Dati IBE/{ibe_b_name}.json", f"Dati/Dati IBE/{ibe_b_name}.params.json")
    if from_date is not None:
        ibe_a_df = ibe_a_df[ibe_a_df.index >= from_date]
        ibe_b_df = ibe_b_df[ibe_b_df.index >= from_date]
    if to_date is not None:
        ibe_a_df = ibe_a_df[ibe_a_df.index <= to_date]
        ibe_b_df = ibe_b_df[ibe_b_df.index <= to_date]
    if ibe_a_df.empty:
        print(f"There is no data for {ibe_a_name} in the selected period")
        return
    if ibe_b_df.empty:
        print(f"There is no data for {ibe_b_name} in the selected period")
        return
    ibe_a_df = analysis.outlier_removal(ibe.clip_IBE_data(ibe_a_df))
    ibe_b_df = analysis.outlier_removal(ibe.clip_IBE_data(ibe_b_df))
    if ibe_a_df.empty:
        print(f"There is no data for {ibe_a_name} after outlier remotion")
        return
    if ibe_b_df.empty:
        print(f"There is no data for {ibe_b_name} after outlier remotion")
        return

    if variables is None:
        variables = ["CO2", "PM10", "PM2.5"]
    f, ax = plt.subplots(nrows=3, sharex=True)
    f2, ax2 = plt.subplots(nrows=3, sharex=True)
    i = 0
    for v in variables:
        diff = ibe_a_df[v] - ibe_b_df[v]
        diff.plot(ax=ax[i], linewidth="0.9")
        ax[i].set_title(f"Differenza tra {ibe_a_name} e {ibe_b_name} - {v}")
        if v in units:
            ax[i].set_ylabel(units[v])
        abs_max = diff.abs().max()
        ax[i].set_ylim((-abs_max * 1.1, abs_max * 1.1))
        ax[i].axhline(0, color="red", linestyle="--", linewidth="0.7")
        plt.tight_layout()

        ibe_a_df[v].plot(ax=ax2[i], linewidth="0.9", label=ibe_a_name)
        ibe_b_df[v].plot(ax=ax2[i], linewidth="0.9", label=ibe_b_name)
        ax2[i].set_title(v)
        if v in units:
            ax2[i].set_ylabel(units[v])
        ax2[i].legend(prop={'size': 6})
        plt.tight_layout()
        i = i + 1
    plt.show()


if __name__ == '__main__':
    # arpav_ibe_comparison()
    ibe_monthly_analysis("SMART53", "SMART55")
    # ibe_trend()
    # ibe_plot_difference("SMART53", "SMART55", from_date="2020-08-01", to_date="2020-08-31")
    # ibe_plot_difference("SMART53", "SMART55", from_date="2020-12-01")
