import pandas as pd


def read_rain_ARPAV_station(data_filename: str) -> pd.DataFrame:
    """Legge da csv i dati delle precipitazioni giornaliere

    Esempio di file compatibile:

    ISTANTE,Anno,Mese,Giorno,Data,VM

    202001010000,2020,01,01,2020/01/01,0.0

    Parameters
    ----------
    data_filename : str
        Percorso al file .csv

    Returns
    -------
    dp.DataFrame
        Dataframe con i dati delle precipitazioni nella colonna "Precipitation"

    """

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


def read_wind_ARPAV_station(data_filename: str) -> pd.DataFrame:
    """Legge da csv i dati dell'intensità del vento

    Esempio di file compatibile:

    ISTANTE,Anno,Mese,Giorno,Data,VM

    202001010000,2020,01,01,2020/01/01,0.0

    Parameters
    ----------
    data_filename : str
        Percorso al file .csv

    Returns
    -------
    dp.DataFrame
        Dataframe con i dati dell'intensità delvento nella colonna "Wind"

    """

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


def compute_pollutant_dispersion(rain_df: pd.DataFrame, wind_df: pd.DataFrame) -> pd.DataFrame:
    """Calcola le condizioni di dispersione degli inquinanti

    La funzione eseuge un inner join dei due df passati in input e calcola le condizioni di dispersione
    rappresentate in 3 livelli: LOW, MEDIUM e HIGH

    Le condizioni sono state determinate facendo riferimento alla relazione della campagna di
    monitoraggio dell'ARPAV

    Parameters
    ----------
    rain_df : pd.DataFrame
        Dataframe con la colonna "Precipitation" (v. read_rain_ARPAV_station())
    wind_df : pd.DataFrame
        Dataframe con la colonna "Wind" (v. read_wind_ARPAV_station())

    Returns
    -------
    pd.DataFrame
        Dataframe con le condizioni di dispersione nella colonna "Dispersion"

    """

    # merge two dfs
    merge_df = pd.merge(rain_df, wind_df, how="inner", left_index=True, right_index=True)
    merge_df["Dispersion"] = "NA"
    merge_df.loc[(merge_df["Precipitation"] < 1) | (merge_df["Wind"] < 1.5), "Dispersion"] = "LOW"
    merge_df.loc[((merge_df["Precipitation"] >= 1) & (merge_df["Precipitation"] <= 6)) | (
                (merge_df["Wind"] >= 1.5) & (merge_df["Wind"] <= 3)), "Dispersion"] = "MEDIUM"
    merge_df.loc[(merge_df["Precipitation"] > 6) | (merge_df["Wind"] > 3), "Dispersion"] = "HIGH"
    return merge_df
