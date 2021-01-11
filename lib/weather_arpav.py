import pandas as pd


def read_rain_ARPAV_station(data_filename: str):
    df = pd.read_csv(data_filename)
    tz = pd.to_datetime(df["Data"], format="%Y/%m/%d", utc=True)
    del df["ISTANTE"]
    del df["Anno"]
    del df["Mese"]
    del df["Giorno"]
    del df["Data"]
    df["Datetime"] = tz
    df.set_index("Datetime", inplace=True)
    return df.rename(columns={"VM": "Precipitation"})


def read_wind_ARPAV_station(data_filename: str):
    df = pd.read_csv(data_filename)
    tz = pd.to_datetime(df["Data"], format="%Y/%m/%d", utc=True)
    del df["ISTANTE"]
    del df["Anno"]
    del df["Mese"]
    del df["Giorno"]
    del df["Data"]
    df["Datetime"] = tz
    df.set_index("Datetime", inplace=True)
    return df.rename(columns={"VM": "Wind"})


# calcola le condizioni di dispersioni degli inquinanti in base
# all'intensità di precipitazioni e vento.
# Le condizioni sono state determinate dalla relazione
# dell'ARPAV sulla campagna di monitoraggio della qualità
# dell'aria
def compute_pollutant_dispersion(rain_df: pd.DataFrame, wind_df: pd.DataFrame):
    # merge two dfs
    merge_df = pd.merge(rain_df, wind_df, how="inner", left_index=True, right_index=True)
    merge_df["Dispersion"] = "NA"
    merge_df.loc[(merge_df["Precipitation"] < 1) | (merge_df["Wind"] < 1.5), "Dispersion"] = "LOW"
    merge_df.loc[((merge_df["Precipitation"] >= 1) & (merge_df["Precipitation"] <= 6)) | (
                (merge_df["Wind"] >= 1.5) & (merge_df["Wind"] <= 3)), "Dispersion"] = "MEDIUM"
    merge_df.loc[(merge_df["Precipitation"] > 6) | (merge_df["Wind"] > 3), "Dispersion"] = "HIGH"
    return merge_df
