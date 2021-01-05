import ibe
import matplotlib.pyplot as plt
import pandas as pd


def main(sensor_name):
    ibe_df = ibe.read_IBE_sensor(f"{sensor_name}_extended.json", f"{sensor_name}.params.json")
    # ibe_df = ibe_df[ibe_df.index < "2020-08-01"]
    ser = ibe_df.dropna().index.to_series()
    d = ser.diff() > pd.Timedelta(1, "H")
    d = d.apply(lambda x: 1 if x else 0)
    d = d.resample("H").bfill()
    df = pd.DataFrame({"time": d})
    df["shifted"] = df["time"].shift(periods=1)
    df["result"] = 0
    print(ibe_df.head())
    missing_periods = []
    p = []
    for index, row in df.iterrows():
        if row["shifted"] == 0 and row["time"] == 1:
            print("up", index)
            df.loc[index, "result"] = 1
            p.append(index)
        elif row["shifted"] == 1 and row["time"] == 0:
            t = index - pd.Timedelta(1, "H")
            print("down", t)
            df.loc[t, "result"] = 2
            p.append(t)
            missing_periods.append(p)
            p = []
    missing_df = pd.DataFrame(missing_periods, columns=["from", "to"])
    missing_df.to_csv(f"{sensor_name}_missing_periods.csv")
    print(missing_periods)
    # df.plot()
    f, ax = plt.subplots()
    ibe_df.plot(ax=ax, linewidth="0.7")
    for p in missing_periods:
        ax.axvspan(p[0], p[1], color="lightsteelblue")
    ax.legend(prop={'size': 4})
    plt.savefig(f"{sensor_name}_missing_periods.png", dpi=300)


if __name__ == "__main__":
    main("SMART53")
    main("SMART54")
    main("SMART55")
    main("SMART56")
