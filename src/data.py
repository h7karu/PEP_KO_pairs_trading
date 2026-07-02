import yfinance as yf
import numpy as np

def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    # auto_adjust = True ensures that the prices are adjusted for dividends and stock splits
    return data["Close"].dropna()

def get_log_prices(prices):
    return np.log(prices)