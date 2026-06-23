import numpy as np

def sharpe_ratio(daily_pnl):
    if daily_pnl.std() == 0:
        return np.nan
    return daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)

def max_drawdown(cum_pnl):
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return drawdown.min()

def summarize_results(results):
    return {
        "total_pnl": results["cum_pnl"].iloc[-1],
        "sharpe": sharpe_ratio(results["daily_pnl"]),
        "max_drawdown": max_drawdown(results["cum_pnl"]),
        "num_trades": int((results["positions"].diff().fillna(results["positions"]) != 0).sum())
    }