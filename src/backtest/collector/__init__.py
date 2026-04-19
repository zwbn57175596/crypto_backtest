from backtest.collector.binance import BinanceCollector
from backtest.collector.okx import OkxCollector
from backtest.collector.htx import HtxCollector

COLLECTORS = {
    "binance": BinanceCollector,
    "okx": OkxCollector,
    "htx": HtxCollector,
}
