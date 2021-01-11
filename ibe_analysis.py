import pandas as pd
from matplotlib import pyplot as plt
from pandas._libs.tslibs.offsets import MonthEnd
from typing import List, Dict

from lib import weather_arpav, ibe, analysis


def ibe_monthly_analysis(ibe_a_name: str, ibe_b_name: str):
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
    analysis.bar_plot_resampled(smart53_df, freq, show=True, name="SMART53",
                                variables=["NO2", "O3", "PM10", "PM2.5"])


def ibe_plot_difference(ibe_a_name: str, ibe_b_name: str, variables: List[str] = None, units: Dict[str, str] = None,
                        from_date: str = None, to_date: str = None):
    """plotta il grafico delle differenze tra due sensori IBIMET.
    è possibile indicare le variabili da plottare come lista di stringhe
    e le loro unità di misura come dizionario
    è possibile anche limitare il periodo di analisi dei dati passando
    a from_date e to_date delle stringhe in formato 'AAAA-MM-GG'"""
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


if __name__ == "__main__":
    ibe_monthly_analysis("SMART53", "SMART55")
    # ibe_trend()
    # ibe_plot_difference("SMART53", "SMART55", from_date="2020-08-01", to_date="2020-08-31")
    # ibe_plot_difference("SMART53", "SMART55", from_date="2020-12-01")
