import pandas as pd


def run_backtest(
    prices,
    positions,
    hedge_ratios,
    y_col="PEP",
    x_col="KO",
    tc_rate=0.0005,
    slippage_rate=0.0005, #these costs are considered conservative
):
    positions = positions.fillna(0)

    beta = hedge_ratios["beta"].reindex(prices.index).fillna(0)

    y_price = prices[y_col].reindex(positions.index)
    x_price = prices[x_col].reindex(positions.index)

    y_units = positions
    x_units = -beta * positions if beta is not None else 0

    delta_y = y_units.diff().fillna(y_units)
    delta_x = x_units.diff().fillna(x_units)

    trade_notional = (
        delta_y.abs() * y_price
        + delta_x.abs() * x_price
    )

    trading_costs = trade_notional * (tc_rate + slippage_rate)

    y_prev = y_units.shift(1).fillna(0)
    x_prev = x_units.shift(1).fillna(0)

    daily_pnl = (
        y_prev * y_price.diff().fillna(0)
        + x_prev * x_price.diff().fillna(0)
    )

    daily_pnl = daily_pnl - trading_costs
    cum_pnl = daily_pnl.cumsum()

    results = pd.DataFrame({
        "positions": positions,
        "daily_pnl": daily_pnl,
        "cum_pnl": cum_pnl,
        "trading_costs": trading_costs,
        "y_units": y_units,
        "x_units": x_units,
    })

    return results