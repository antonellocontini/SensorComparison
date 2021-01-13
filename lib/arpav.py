import pandas as pd


def read_ARPAV_station(data_filename: str) -> pd.DataFrame:
    """Legge da .csv i dati della stazione mobile dell'ARPAV

    Parameters
    ----------
    data_filename : str
        Percorso al file .csv da leggere. Il file deve avere il seguente formato:

        Datetime,CO,DVP,NO,NO2,NOx,O3,SO2,T,RH,VVP

        17/06/2020 17,0.116,275,2,4,5,92,3,23,55,0.7

    Returns
    -------
    pd.DataFrame
        Dataframe con i dati del .csv
    """
    df = pd.read_csv(data_filename)
    # df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H", utc=True)

    # leggo timestamp in UTC+2 (CEST) e converto in UTC
    # N.B. faccio questo perchè c'è un bug nel plot delle timeseries dove la timezone
    # non viene considerata
    tz_naive = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H")
    df["Datetime"] = tz_naive.dt.tz_localize(tz="Europe/Rome").dt.tz_convert(tz="UTC")
    df.set_index("Datetime", inplace=True)
    return df
