"""Routine per leggere dati da cloud SGL

"""

import requests
import numpy as np
import datetime
import pandas as pd
import scipy.stats as stats
from typing import Dict, Union


def query_ensor(sensorURI, fromTime, toTime, valueName):
    """Query al cloud SGL

    Parameters
    ----------
    sensorURI : str
        URI del sensore sul cloud SGL
    fromTime : str
        stringa indicante il numero di giorni da scaricare es. '4-day'
    toTime : str
        Data e ora in formato ISO es. '2020-12-01T00:00:00'
    valueName : str
        valueName del parametro da scaricare

    Returns
    -------
    Dict

    """

    s = f"https://smartgardalake.snap4.eu/ServiceMap/api/v1/?serviceUri={sensorURI}&fromTime={fromTime}&toTime={toTime}&valueName={valueName}"
    print(s)
    response = requests.get(s)
    data = response.json()
    values = []
    try:
        values = data["realtime"]["results"]["bindings"]
    except KeyError:
        print("[WARN] empty dataset")
    values.reverse()
    result = {
        "measuredTime": [],
        valueName: [],
    }
    print(len(values))
    for i in range(len(values)):
        v = values[i]
        result["measuredTime"].append(v["measuredTime"]["value"])
        try:
            float_measure = float(v[valueName]["value"])
            if valueName == "CO2" and float_measure > 2000:
                result[valueName].append(np.nan)
            else:
                result[valueName].append(float_measure)
        except ValueError:
            result[valueName].append(np.nan)
    return result


def multiday_query(sensorURI, fromTime, toTime, valueName):
    """Wrapper per richiedere periodi temporali più estesi di un giorno senza perdere risoluzione temporale

    Parameters
    ----------
    sensorURI : str
    fromTime : str
    toTime : str
    valueName : str

    Returns
    -------
    Dict

    """

    result = None
    l = fromTime.split("-")
    nDays = int(l[0])
    period = l[1]
    if period != "day":
        return query_ensor(sensorURI, fromTime, toTime, valueName)
    for i in range(nDays):
        temp = query_ensor(sensorURI, "1-day", toTime, valueName)
        if result is None:
            result = temp
        else:
            result["measuredTime"] = temp["measuredTime"] + result["measuredTime"]
            result[valueName] = temp[valueName] + result[valueName]
        toTime = datetime.datetime.strftime(
            datetime.datetime.strptime(toTime, "%Y-%m-%dT%H:%M:%S") - datetime.timedelta(days=1),
            "%Y-%m-%dT%H:%M:%S")
    return result


def get_traffic_sensor_df(sensorURI: str, fromTime: str, toTime: str, resampleFreq: str = None, remove_outliers=False):
    """Query a SGL di un sensore del traffico

    Vedi query_ensor() per sensorURI, fromTime e toTime

    Parameters
    ----------
    sensorURI : str
    fromTime : str
    toTime : str
    resampleFreq : str
    remove_outliers : bool

    Returns
    -------
    pd.DataFrame

    """
    values = ["count", "sumSpeed"]
    result = None
    for v in values:
        # data = query_ensor(sensorURI, fromTime, toTime, v)
        data = multiday_query(sensorURI, fromTime, toTime, v)
        df = pd.DataFrame(data, columns=["measuredTime", v])
        df["measuredTime"] = pd.to_datetime(df["measuredTime"])
        df.index = df["measuredTime"]
        del df["measuredTime"]
        if remove_outliers:
            z_scores = np.abs(stats.zscore(df))
            print(f"Removed outliers: {df.size - df[(z_scores < 3).all(axis=1)].size}")
            df = df[(z_scores < 3).all(axis=1)]
        if resampleFreq is not None:
            df = df.resample(resampleFreq).sum()
        if result is not None:
            result = pd.merge_ordered(result, df, left_on="measuredTime", right_on="measuredTime")
            result.index = result["measuredTime"]
            del result["measuredTime"]
        else:
            result = df
    # avg speed
    result["avgSpeed"] = result["sumSpeed"] / result["count"]
    result.loc[~np.isfinite(result["avgSpeed"]), "avgSpeed"] = np.nan
    result["avgSpeed"] = result["avgSpeed"].interpolate()
    return result


def read_traffic_sensor_from_csv(path: str) -> pd.DataFrame:
    """Leggi dati del traffico da .csv, il file dev'essere nel formato letto da SGL

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
    """

    df = pd.read_csv(path)
    df["measuredTime"] = pd.to_datetime(df["measuredTime"])
    df.set_index("measuredTime", inplace=True)
    return df
