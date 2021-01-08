import pandas as pd


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
