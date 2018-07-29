# Deep analysis w/ cryptocurrency

https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-importerror-cannot-import-name-is-list-like

![python](https://forthebadge.com/images/badges/made-with-python.svg "python")

## Build setup for deep-analysis

<!-- https://blog.patricktriest.com/analyzing-cryptocurrencies-python/ -->

```sh
git clone https://github.com/Draichi/deep-analysis.git

cd deep-analysis

pip3 install -r requeriments.txt

python3 cryptos/deep_analysis.py
# a window will pop-up with the chart

python3 cryptos/cryptos_prediction.py -d 7 -c 0.02 -$ BTC
# or python3 cryptos/cryptos_prediction.py --help

```

