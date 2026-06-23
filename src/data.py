import yfinance as yf
import numpy as np

def download_prices(tickers, start, end=None):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return data["Close"].dropna()

def get_log_prices(prices):
    return np.log(prices)