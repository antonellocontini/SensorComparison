# SensorComparison

Mini-libreria per l'analisi e il confronto dei dati provenienti dai sensori
IBE e ARPAV.
Usa le librerie Pandas, scipy, scikit-learn e matplotlib su Python 3.8

## Come usarla
Nella cartella `lib` sono inclusi metodi per leggere i dataset presenti
in questo repository. Inoltre nel modulo `analysis` sono presenti
alcuni metodi per facilitare il confronto tra i diversi dataset.

Il modulo `main` fa uso di questi metodi per visualizzare un confronto
tra i sensori IBE e la centralina mobile dell'ARPAV.

## Dati presenti
Sono presenti sia i dati della centralina mobile ARPAV
che i dati dei sensori IBE.
Inoltre sono stati aggiunti i dati meteo (intensità vento e
precipitazioni) raccolti dalla stazione ARPAV di Bardolino.

### Centralina mobile ARPAV
Nel file MMC.csv sono presenti i dati dal 17/06/2020 ore 17
al 27/07/2020 ore 10, periodo nel quale la centralina era dislocata
a Garda

Il file MMC_2020.xlsx contiene i dati di tutto l'anno

Il file MMC.ods è stato utilizzato per convertire i dati prima
di passarli in csv

### Sensori IBE (smartea)
I file SMART53.json fino a SMART56.json contengono i dati
fino al 21/12/2020

I sensori SMART53 (Lazise - dogana veneta) e
SMART54 (torri del benaco - incrocio con via albisano)
hanno dati a partire dal 01/07/2020.
Per gli altri due mancano i dati nel periodo della centralina ARPAV

Nello script i dati IBE vengono ricampionati con frequenza
oraria per allinearli ai dati ARPAV

### Stazione metereologica ARPAV Bardolino
La stazione raccoglie con frequenza giornaliera dati sulle condizioni
meteorologiche. In questa repository sono contenuti solamente i dati
relativi a precipitazioni e intensità del vento, in quanto sono ritenuti
da parte dell'ARPAV rappresentativi anche delle condizioni su Garda.

## Variabili sensori e unità di misura
### variabili comuni
* CO: mg/m3
* NO2: µg/m3
* O3: µg/m3
* T: C°
* RH (relative humidityi): %

### solo centralina mobile ARPAV
* NO: µg/m3
* NOx: µg/m3
* SO2: µg/m3
* VVP: m/s

### solo sensori IBE (smartea)
* CO2: ppm
* PM10: µg/m3
* PM2.5: µg/m3
* NO: non calibrato
