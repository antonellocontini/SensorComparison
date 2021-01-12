from lib import sgl


def main():
    df = sgl.getTrafficSensorDf("http://www.disit.org/km4city/resource/iot/iotobsf/GardaLake/Gardesana-BCC-Probe1",
                                "140-day", "2021-01-12T00:00:00")
    df.to_csv("Dati/Dati FLOUD/BCC-Probe1.csv")


if __name__ == "__main__":
    main()
