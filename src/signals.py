import numpy as np
import pandas as pd
import statsmodels.api as sm


def rolling_hedge_ratio(log_prices, y_col="PEP", x_col="KO", window=60):
    alphas = []
    betas = []

    for i in range(len(log_prices)):
        if i < window:
            alphas.append(np.nan)
            betas.append(np.nan)
            continue

        y = log_prices[y_col].iloc[i - window:i]
        x = log_prices[x_col].iloc[i - window:i]

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        alphas.append(model.params["const"])
        betas.append(model.params[x_col])

    return pd.DataFrame(
        {
            "alpha": alphas,
            "beta": betas,
        },
        index=log_prices.index,
    )


def compute_rolling_spread(log_prices, hedge_ratios, y_col="PEP", x_col="KO"):
    spread = (
        log_prices[y_col]
        - (hedge_ratios["alpha"] + hedge_ratios["beta"] * log_prices[x_col])
    )

    return spread


def compute_rolling_zscore(spread, window=60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    zscore = (spread - rolling_mean) / rolling_std
    return zscore


def generate_positions(zscore, entry=2.0, exit=0.5):
    positions = []
    pos = 0

    for z in zscore:
        if pd.isna(z):
            positions.append(0)
            continue

        if pos == 0:
            if z < -entry:
                pos = 1
            elif z > entry:
                pos = -1

        elif pos == 1:
            if z > -exit:
                pos = 0

        elif pos == -1:
            if z < exit:
                pos = 0

        positions.append(pos)

    return pd.Series(positions, index=zscore.index, name="position")