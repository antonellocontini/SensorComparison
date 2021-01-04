import json

import pandas as pd


def convert_IBE_json_to_df(js, calibration_params):
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


def read_IBE_sensor(data_filename, params_filename):
    with open(params_filename) as f:
        calibration_params = json.load(f)
    with open(data_filename) as f:
        js = json.load(f)
    df = convert_IBE_json_to_df(js, calibration_params)
    df = df.resample("H").mean()
    return df


# rimuove i valori che superano i limiti massimi rilevabili
# dai sensori IBE
def clip_IBE_data(df, limits=None):
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
    return copy_df