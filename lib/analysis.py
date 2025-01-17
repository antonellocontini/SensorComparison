import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from pathlib import Path
import datetime as dt
from typing import List, Dict, Union

from lib import arpav
from lib.ibe import read_IBE_sensor


def outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove gli outlier da un dataframe dato input

    Parameters
    ----------
    df : pd.DataFrame
        Il dataframe su cui rimuovere gli outliers

    Returns
    -------
    pd.DataFrame
        Il dataframe con gli outliers rimossi
    """

    for col in df:
        z_score = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        df[col][z_score >= 3] = np.nan
    return df


def plot_pollutant_dispersion(weather_df: pd.DataFrame, ax=None, plot_low=True, plot_medium=True, plot_high=True,
                              show=False):
    """Sovraimpone su un plot la visualizzazione della dispersione degli inquinanti

    Parameters
    ----------
    weather_df : pd.DataFrame
        Un dataframe contenente la colonna "Dispersion" (v. compute_pollutant_dispersion())
    ax
    plot_low : bool
        True per plottare le condizioni di bassa dispersione
    plot_medium : bool
        True per plottare le condizioni di media dispersione
    plot_high : bool
        True per plottare le condizioni di alta dispersione
    show : bool
        True per visualizzare subito il plot

    Returns
    -------

    """

    if ax is None:
        _, ax = plt.subplots()
    for index, row in weather_df.iterrows():
        if row["Dispersion"] == "LOW" and plot_low:
            ax.axvspan(index, index + pd.Timedelta(days=1), color="mistyrose")
        elif row["Dispersion"] == "MEDIUM" and plot_medium:
            ax.axvspan(index, index + pd.Timedelta(days=1), color="lightyellow")
        elif row["Dispersion"] == "HIGH" and plot_high:
            ax.axvspan(index, index + pd.Timedelta(days=1), color="honeydew")
    if show:
        plt.show()


def similarity_common_variables(reference_df: pd.DataFrame, test_df: pd.DataFrame, reference_name: str, test_name: str,
                                variables: List[str] = None, units: Dict[str, str] = None, window=24,
                                save_graphs=True, show=True, folder: str = None, weather_df: pd.DataFrame = None):
    """Confronta due dataframe producendo per ogni colonna plot di RMSE, NRMSE e indice di correlazione,
    sia su tutte le righe che in finestra mobile.

    I due dataframe devono avere la stessa frequenza temporale (v. df.resample()).

    Parameters
    ----------
    reference_df : pd.DataFrame
        Questo dataframe viene preso come riferimento per la normalizzazione dell'RMSE
    test_df : pd.DataFrame
        Secondo dataframe
    reference_name : str
        Nome da dare nei plot al dataframe di riferimento
    test_name : str
        Nome da dare nei plot al dataframe di test
    variables : List[str]
        Elenco delle colonne da analizzare, devono essere presenti in entrambi i dataframe
    units : Dict[str, str]
        Dizionario con le unità da misura per ogni colonna, visualizzate nei plot
    window : int
        Dimensione della finestra mobile
    save_graphs : bool
        Se True salva i plot su disco
    show : bool
        Se True mostra i plot su schermo
    folder : str
        Directory nella quale salvare i plot
    weather_df : pd.DataFrame
        Se specificato, mostra all'interno dei plot le condizioni di dispersione degli inquinanti.
        Il dataframe deve avere una colonna denominata "Dispersion" (v. compute_pollutant_dispersion())

    Returns
    -------
    List[Union[pd.DataFrame, pd.Series, None]]
        Un array di 4 elementi [mov_r, mov_rmse, r, rmse, nrmse]:
        Ogni elemento è un dataframe con le statistiche per ogni variabile
        mov_r e mov_rmse contengono una colonna per variabile con la statistica
        calcolata in moving window.
        r, rmse e nrmse sono composti di una sola colonna, una riga per variabile
        e ogni valore corrisponde al valore della statistica per quella variabile.
    """

    def pearson(ser):
        try:
            reference_ser = reference_df[v][ser.index].dropna()
            test_ser = test_df[v][ser.index].dropna()
            if reference_ser.nunique() <= 1 or test_ser.nunique() <= 1:
                return np.nan
            r, p = stats.pearsonr(reference_ser, test_ser)
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    def rmse(ser):
        try:
            # RMSE (squared=False)
            # sklearn calcola MSE = sum_i^n((arpav[i] - ibe[i])**2)/n
            r = mean_squared_error(reference_df[v][ser.index].dropna(),
                                   test_df[v][ser.index].dropna(), squared=False)
            return r
        except (KeyError, ValueError) as e:
            return np.nan

    def nrmse(ser):
        min_value = reference_df[v].min()
        max_value = reference_df[v].max()
        try:
            # RMSE (squared=False)
            r = mean_squared_error(reference_df[v][ser.index].dropna(),
                                   test_df[v][ser.index].dropna(), squared=False)
            nr = r * 100 / (max_value - min_value)
            return nr
        except (KeyError, ValueError) as e:
            return np.nan

    if units is None:
        units = {
            "NO2": "µg/m3",
            "O3": "µg/m3",
            "CO": "mg/m3",
            "T": "C°",
            "RH": "%"
        }

    if variables is None:
        variables = ["NO2", "O3", "CO", "T", "RH"]

    if folder is None:
        folder = "similarity_graphs"

    merge_df = pd.merge(reference_df.add_prefix(f"{reference_name} "), test_df.add_prefix(f"{test_name} "), how="inner",
                        left_index=True, right_index=True)
    for v in variables:
        f_boxplot, ax_boxplot = plt.subplots()
        merge_df.boxplot(ax=ax_boxplot, column=[f"{reference_name} {v}", f"{test_name} {v}"])
        if save_graphs:
            graph_directory = Path(f"{folder}/{reference_name} v {test_name}")
            graph_directory.mkdir(parents=True, exist_ok=True)
            graph_filename = graph_directory.joinpath(f"{v}_boxplot.png")
            plt.savefig(graph_filename, dpi=300)

    pearson_rs = {}
    nrmses = {}
    rmses = {}
    mov_pearsons = []
    mov_rmses = []
    for v in variables:
        if v in reference_df and v in test_df:
            ref_v_name = f"REFERENCE {v}"
            test_v_name = f"TEST {v}"
            unique_df = pd.concat([reference_df[v].rename(ref_v_name), test_df[v].rename(test_v_name)], axis=1).dropna()
            f, ax = plt.subplots(nrows=3, sharex=True)
            pearson_ax = ax[0]
            rmse_ax = ax[1]
            data_ax = ax[2]
            # print(v)
            rol = reference_df[v].rolling(window=window)

            v_pearson_ser = rol.apply(pearson, raw=False)
            v_pearson_ser.rename(v, inplace=True)
            mov_pearsons.append(v_pearson_ser)
            v_pearson_ser.plot(ax=pearson_ax)
            overall_r, overall_p = stats.pearsonr(unique_df[ref_v_name], unique_df[test_v_name])
            overall_r = overall_r
            pearson_rs[v] = overall_r
            pearson_ax.set_ylim([-1, 1])
            pearson_ax.axhline(overall_r, color="red", linestyle="--")
            pearson_ax.set_title(f"{v} - Moving window pearson R - R complessivo: {overall_r:.3f}")

            rmse_ser = rol.apply(rmse, raw=False)
            rmse_ser.rename(v, inplace=True)
            mov_rmses.append(rmse_ser)
            # rmse_avg = rmse_ser.mean()
            overall_rmse = mean_squared_error(unique_df[ref_v_name], unique_df[test_v_name], squared=False)
            if unique_df[ref_v_name].max() - unique_df[ref_v_name].min() > 0:
                nrmses[v] = overall_rmse / (unique_df[ref_v_name].max() - unique_df[ref_v_name].min())
                rmses[v] = overall_rmse
            rmse_ser.plot(ax=rmse_ax)
            rmse_ax.axhline(overall_rmse, color="red", linestyle="--")
            rmse_ax.set_title(f"{v} - Moving window RMSE - RMSE complessivo: {overall_rmse:.3f}")
            if v in units:
                rmse_ax.set_ylabel(units[v])

            # nrmse_ser = rol.apply(nrmse, raw=False)
            # overall_nrmse = mean_squared_error(unique_df[f"ARPAV {v}"], unique_df[f"IBE {v}"], squared=False) / (unique_df)
            # nrmse_avg = nrmse_ser.mean()
            # nrmse_ser.plot(ax=ax[2])
            # ax[2].set_title(f"Mov. window Normalized RMSE - {v} Average NRMSE: {nrmse_avg:.3f}%")
            # ax[2].set_ylabel("%")

            reference_df[v].plot(ax=data_ax, label=f"{reference_name} {v}")
            test_df[v].plot(ax=data_ax, label=f"{test_name} {v}")
            if weather_df is not None:
                plot_pollutant_dispersion(weather_df, ax=data_ax)
            data_ax.set_title(f"Confronto tra {reference_name} e {test_name}")
            if v in units:
                data_ax.set_ylabel(units[v])
            data_ax.legend(prop={'size': 6})
            plt.tight_layout()
            if save_graphs:
                graph_directory = Path(f"{folder}/{reference_name} v {test_name}")
                graph_directory.mkdir(parents=True, exist_ok=True)
                graph_filename = graph_directory.joinpath(f"{v}.png")
                plt.savefig(graph_filename, dpi=300)

    if len(pearson_rs) > 0:
        summary_f, summary_ax = plt.subplots()
        pearson_df = pd.DataFrame.from_dict(pearson_rs, orient="index", columns=["Pearson"])
        pearson_df.plot.bar(ax=summary_ax)
        summary_ax.set_ylim([-1, 1])
        summary_ax.set_title(f"Correlazione tra {reference_name} e {test_name}")
        for p in summary_ax.patches:
            h = p.get_height()
            if h > 0:
                label_y = -0.12
            else:
                label_y = 0.05
            summary_ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., label_y), ha="center")
        if save_graphs:
            graph_directory = Path(f"{folder}/{reference_name} v {test_name}")
            graph_directory.mkdir(parents=True, exist_ok=True)
            graph_filename = graph_directory.joinpath(f"summary.png")
            plt.savefig(graph_filename, dpi=300)
    else:
        pearson_df = None

    if len(nrmses) > 0:
        nrmse_f, nrmse_ax = plt.subplots()
        nrmse_df = pd.DataFrame.from_dict(nrmses, orient="index", columns=["NRMSE"])
        nrmse_df.plot.bar(ax=nrmse_ax)
        nrmse_ax.set_ylim([0, None])
        nrmse_ax.set_title(f"RMSE normalizzato tra {reference_name} e {test_name}")
        for p in nrmse_ax.patches:
            h = p.get_height()
            label_y = h + 0.05
            nrmse_ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., label_y), ha="center")
        if save_graphs:
            graph_directory = Path(f"{folder}/{reference_name} v {test_name}")
            graph_directory.mkdir(parents=True, exist_ok=True)
            graph_filename = graph_directory.joinpath(f"nrmse.png")
            plt.savefig(graph_filename, dpi=300)
    else:
        nrmse_df = None

    if len(rmses) > 0:
        rmse_df = pd.DataFrame.from_dict(rmses, orient="index", columns=["RMSE"])
    else:
        rmse_df = None

    if len(mov_rmses) > 0:
        mov_rmse_df = pd.concat(mov_rmses, axis=1)
    else:
        mov_rmse_df = None

    if len(mov_pearsons) > 0:
        mov_pearsons_df = pd.concat(mov_pearsons, axis=1)
    else:
        mov_pearsons_df = None

    if show:
        plt.show()

    return [mov_pearsons_df, mov_rmse_df, pearson_df, rmse_df, nrmse_df]


def plot_common_variables(arpav_df: pd.DataFrame, ibe_df: pd.DataFrame, show=False):
    # f, ax = plt.subplots(2, 3)
    variables = [["NO2", "O3", "CO"], ["T", "RH", None]]
    i = 0
    for l in variables:
        j = 0
        for v in l:
            if v is not None:
                f2, ax2 = plt.subplots()
                merge_df = pd.merge(arpav_df[v], ibe_df[v], left_index=True, right_index=True, how="outer")
                merge_df.plot(ax=ax2)
                plt.tight_layout()
            j = j + 1
        i = i + 1
    if show:
        plt.show()


# Visualizza il dataframe dato in input come grafico a barre,
# dopo aver ricampionato alla frequenza specificata
# Si può specificare un sottoinsieme delle colonne del dataframe
# passando nel parametro variables una lista di stringhe
def bar_plot_resampled(df: pd.DataFrame, freq: Union[str, pd.tseries.offsets.DateOffset], name=None,
                       variables: List[str] = None,
                       show=False):
    """Visualizza il dataframe dato in input come grafico a barre, dopo aver ricampionato alla frequenza specificata

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe con i dati da visualizzare
    freq : Union[str, pd.tseries.offsets.DateOffset]
        Per ora supporta solamente offset mensili e settimanali
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    name : str
        Titolo da dare al plot
    variables : List[str]
        Elenco delle colonne da plottare (se non specificato verranno plottate tutte)
    show : bool
        settato a True mostra a schermo il plot

    Returns
    -------

    """
    if variables is not None:
        copy_df = df[variables].resample(freq).mean()
    else:
        copy_df = df.resample(freq).mean()

    f, ax = plt.subplots()
    copy_df.plot.bar(ax=ax)
    ax.legend(prop={'size': 6})
    if freq == "MS" or freq == "M":
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(copy_df.index.month_name()))
    if freq == "W" or type(freq) is pd.tseries.offsets.Week:
        ax.xaxis.set_major_formatter(
            ticker.FixedFormatter(copy_df.index.to_series().apply(lambda x: dt.datetime.strftime(x, "%d/%m"))))
    if name is not None:
        ax.set_title(name)
    plt.tight_layout()
    # for col in copy_df:
    #     f, ax = plt.subplots()
    #     copy_df[col].plot.bar(ax=ax)
    #     plt.tight_layout()

    if show:
        plt.show()


def old_main():
    from_date = "2020-07-01"
    to_date = "2020-07-27"
    smart53_df = read_IBE_sensor(f"../Dati/Dati IBE/SMART53.json", f"Dati/Dati IBE/SMART53.params.json")
    restricted_smart53_df = smart53_df[(smart53_df.index > from_date) & (smart53_df.index < to_date)]
    smart54_df = read_IBE_sensor(f"../Dati/Dati IBE/SMART54.json", f"Dati/Dati IBE/SMART54.params.json")
    restricted_smart54_df = smart54_df[(smart54_df.index > from_date) & (smart54_df.index < to_date)]

    # remove outliers
    restricted_smart53_df = outlier_removal(restricted_smart53_df)
    restricted_smart54_df = outlier_removal(restricted_smart54_df)

    arpav_df = arpav.read_ARPAV_station("../Dati/Unità mobile - Garda/MMC.csv")
    restricted_arpav_df = arpav_df[(arpav_df.index > from_date) & (arpav_df.index < to_date)]
    # plot_common_variables(restricted_arpav_df, restricted_smart53_df, True)
    # similarity_common_variables(restricted_arpav_df, restricted_smart53_df, "ARPAV", "SMART53", show=False)
    # similarity_common_variables(restricted_arpav_df, restricted_smart54_df, "ARPAV", "SMART54", show=False)
    # similarity_common_variables(restricted_smart53_df, restricted_smart54_df, "SMART53", "SMART54", show=False,
    #                             variables=["NO2", "O3", "CO", "T", "RH", "CO2", "PM10", "PM2.5"])
    for month in range(7, 13, 2):
        from_date = pd.to_datetime(f"2020-{month:02}-01", utc=True)
        to_date = pd.to_datetime(f"2020-{month:02}-01", utc=True) + MonthEnd(2)
        restricted_smart53_df = smart53_df[(smart53_df.index > from_date) & (smart53_df.index < to_date)]
        restricted_smart54_df = smart54_df[(smart54_df.index > from_date) & (smart54_df.index < to_date)]
        restricted_smart53_df["CO2"][restricted_smart53_df["CO2"] > 1000.0] = 1000.0
        restricted_smart54_df["CO2"][restricted_smart54_df["CO2"] > 1000.0] = 1000.0
        restricted_smart53_df = outlier_removal(restricted_smart53_df)
        restricted_smart54_df = outlier_removal(restricted_smart54_df)
        similarity_common_variables(restricted_smart53_df, restricted_smart54_df, "SMART53", "SMART54", show=False,
                                    variables=["NO2", "O3", "CO", "T", "RH", "CO2", "PM10", "PM2.5"],
                                    folder=f"IBE comparison - 2020-{month:02}")

# if __name__ == "__main__":
#    old_main()
