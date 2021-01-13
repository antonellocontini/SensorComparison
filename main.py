from lib import analysis, arpav, ibe, weather_arpav
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def arpav_ibe_comparison():
    """Confronto dei dati IBE con la stazione mobile ARPAV nel mese di Luglio 2020

    Returns
    -------

    """
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


if __name__ == '__main__':
    arpav_ibe_comparison()
