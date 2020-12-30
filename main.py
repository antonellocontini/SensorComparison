import analysis
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt


def arpav_ibe_comparison():
    from_date = "2020-07-01"
    to_date = "2020-07-27"
    smart53_df = analysis.read_IBE_sensor(f"SMART53.json", f"SMART53.params.json")
    restricted_smart53_df = smart53_df[(smart53_df.index > from_date) & (smart53_df.index < to_date)]
    smart54_df = analysis.read_IBE_sensor(f"SMART54.json", f"SMART54.params.json")
    restricted_smart54_df = smart54_df[(smart54_df.index > from_date) & (smart54_df.index < to_date)]

    # remove outliers
    restricted_smart53_df = analysis.outlier_removal(restricted_smart53_df)
    restricted_smart54_df = analysis.outlier_removal(restricted_smart54_df)

    arpav_df = analysis.read_ARPAV_station("MMC.csv")
    restricted_arpav_df = arpav_df[(arpav_df.index > from_date) & (arpav_df.index < to_date)]
    analysis.similarity_common_variables(restricted_arpav_df, restricted_smart53_df, "ARPAV", "SMART53", show=False)
    analysis.similarity_common_variables(restricted_arpav_df, restricted_smart54_df, "ARPAV", "SMART54", show=False)


def ibe_monthly_analysis():
    smart53_df = analysis.read_IBE_sensor(f"SMART53.json", f"SMART53.params.json")
    smart54_df = analysis.read_IBE_sensor(f"SMART54.json", f"SMART54.params.json")
    units = {
        "NO2": "µg/m3",
        "O3": "µg/m3",
        "CO": "mg/m3",
        "T": "C°",
        "RH": "%",
        "PM10": "µg/m3",
        "PM2.5": "µg/m3"
    }
    limits = {
        "O3": 300,
        "NO2": 300,
        "CO": 30,
        "PM2.5": 300,
        "PM10": 300,
        "CO2": 1000
    }
    for month in range(7, 13, 2):
        from_date = pd.to_datetime(f"2020-{month:02}-01", utc=True)
        to_date = pd.to_datetime(f"2020-{month:02}-01", utc=True) + MonthEnd(2)
        restricted_smart53_df = smart53_df[(smart53_df.index > from_date) & (smart53_df.index < to_date)]
        restricted_smart54_df = smart54_df[(smart54_df.index > from_date) & (smart54_df.index < to_date)]
        for v in limits:
            restricted_smart53_df[v][restricted_smart53_df[v] > limits[v]] = limits[v]
            restricted_smart54_df[v][restricted_smart54_df[v] > limits[v]] = limits[v]
        restricted_smart53_df = analysis.outlier_removal(restricted_smart53_df)
        restricted_smart54_df = analysis.outlier_removal(restricted_smart54_df)
        analysis.similarity_common_variables(restricted_smart53_df, restricted_smart54_df, "SMART53", "SMART54",
                                             show=False,
                                             variables=["NO2", "O3", "CO", "T", "RH", "CO2", "PM10", "PM2.5"],
                                             units=units,
                                             folder=f"IBE comparison - 2020-{month:02}")
        plt.close("all")


if __name__ == '__main__':
    ibe_monthly_analysis()
