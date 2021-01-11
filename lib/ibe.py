import json
import pandas as pd
from typing import Dict, Union, List


def convert_IBE_json_to_df(js: Dict[str, Union[str, List[Dict[str, Union[str, float]]]]],
                           calibration_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Converte il JSON con i dati di un sensore IBE in un dataframe

    Parameters
    ----------
    js : Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]
        JSON con i dati da convertire
    calibration_params : Dict[str, Dict[str, float]]
        Dizionario con i parametri di calibrazione
    Returns
    -------
    pd.DataFrame
        Dataframe con i dati convertiti
    """

    df = pd.DataFrame.from_dict(js["data"])
    columns = ['y_coord', 'RH', 'PM10', 'CO2', 'PM2.5', 'x_coord', 'O3', 'VOC',
               'NO_A', 'T', 'NO2', 'CO', 'NO2_A']
    for c in columns:
        df[c] = df[c].astype("float64")
    for v in calibration_params:
        params = calibration_params[v]
        df[v] = params["a"] * df[v] + \
                params["b"] * df["T"] + \
                params["c"] * df[v] * df["T"] + \
                params["q"]
    df["data"] = pd.to_datetime(df["data"], utc=True)
    df.set_index("data", inplace=True)
    return df


def read_IBE_sensor(data_filename: str, params_filename: str) -> pd.DataFrame:
    """Legge da file i dati IBE e i parametri di calibrazione e ritorna un dataframe.
    I dati vengono ricampionati ad una frequenza orario facendo la media

    Parameters
    ----------
    data_filename : str
        Percorso al JSON con i dati
    params_filename : str
        Percorso al JSON con i parametri
    Returns
    -------
    pd.DataFrame
        Dataframe con i dati letti
    """
    with open(params_filename) as f:
        calibration_params = json.load(f)
    with open(data_filename) as f:
        js = json.load(f)
    df = convert_IBE_json_to_df(js, calibration_params)
    df = df.resample("H").mean()
    return df


def clip_IBE_data(df: pd.DataFrame, limits: Dict[str, float] = None) -> pd.DataFrame:
    """rimuove i valori che superano i limiti massimi rilevabili dai sensori IBE

    Parameters
    ----------
    df : pd.DataFrame
    limits : Dict[str, float]

    Returns
    -------
    pd.DataFrame
        Dataframe clippato

    """

    copy_df = df.copy()
    if limits is None:
        limits = {
            "O3": 300,
            "NO2": 300,
            "CO": 30,
            "PM2.5": 300,
            "PM10": 300,
            "CO2": 1000
        }
    for v in limits:
        copy_df[v][copy_df[v] > limits[v]] = limits[v]
        copy_df[v][copy_df[v] < 0.0] = 0.0
    return copy_df
