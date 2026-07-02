import numpy as np
import pandas as pd
import statsmodels.api as sm


# kalman filter for dynamic hedge ratio

def kalman_hedge_ratio(log_prices, y_col="PEP", x_col="KO", delta=1e-4, R_var=1e-2):
    """
    Estimate a time-varying hedge ratio using a Kalman filter.

    State vector: [beta, intercept]
    Observation:  y_t = beta_t * x_t + intercept_t + noise

    delta : process noise scalar — controls how fast beta is allowed to drift.
            Higher = adapts faster but noisier. For stable pairs like KO/PEP,
            1e-4 to 1e-5 is usually appropriate.
    R_var : observation noise variance. Higher = filter trusts the model more
            than new price observations.

    returns DataFrame with columns: beta, intercept, uncertainty (trace of P matrix)
    """
    y = log_prices[y_col].values
    x = log_prices[x_col].values
    n = len(y)

    state = np.zeros(2)       # initial state [beta, intercept] = [0, 0]
    P     = np.eye(2)         # initial uncertainty — high/uninformed
    Q     = delta * np.eye(2) # process noise: how much state drifts per bar

    betas         = np.full(n, np.nan)
    intercepts    = np.full(n, np.nan)
    uncertainties = np.full(n, np.nan)

    for t in range(n):
        F = np.array([x[t], 1.0])   # observation vector [KO_price, 1]

        # ── Predict ──────────────────────────────────────────────
        P_pred = P + Q              # uncertainty grows each bar

        # ── Update ───────────────────────────────────────────────
        innovation = y[t] - F @ state
        S          = F @ P_pred @ F.T + R_var
        K          = P_pred @ F.T / S              # Kalman gain

        state = state + K * innovation
        P     = (np.eye(2) - np.outer(K, F)) @ P_pred

        betas[t]         = state[0]
        intercepts[t]    = state[1]
        uncertainties[t] = np.trace(P)

    return pd.DataFrame(
        {"beta": betas, "intercept": intercepts, "uncertainty": uncertainties},
        index=log_prices.index,
    )

# spread using dynamic hedge ratio

def compute_kalman_spread(log_prices, kalman_ratios, y_col="PEP", x_col="KO"):
    """
    Spread = PEP - (beta_t * KO + intercept_t)

    Uses the Kalman-estimated beta and intercept so the spread adapts
    continuously to slow drift in the relationship — no fixed window.
    """
    spread = (
        log_prices[y_col]
        - (kalman_ratios["beta"] * log_prices[x_col] + kalman_ratios["intercept"])
    )
    return spread.rename("spread")


# OU process — fit params + z-score

def fit_ou_params(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the spread via OLS on the discretised form:

        ΔS_t = a + b * S_{t-1} + ε_t

    where:
        θ (mean-reversion speed) = -b
        μ (long-run mean)        =  a / θ
        σ (noise std)            =  std(ε)
        half_life                =  ln(2) / θ  (in bars)

    Returns
    -------
    dict with keys: theta, mu, sigma, half_life
    Returns None if fit fails or spread is not mean-reverting (b >= 0).
    """
    spread_clean = spread.dropna()
    if len(spread_clean) < 20:
        return None

    delta_s = spread_clean.diff().dropna()
    s_lag   = spread_clean.shift(1).dropna().rename("spread_lag")

    delta_s, s_lag = delta_s.align(s_lag, join="inner")

    X   = sm.add_constant(s_lag)
    res = sm.OLS(delta_s, X).fit()

    b = res.params["spread_lag"]
    a = res.params["const"]

    if b >= 0:
        return None  # not mean-reverting in this window

    theta     = -b
    mu        = a / theta
    sigma     = res.resid.std()
    half_life = np.log(2) / theta

    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life": half_life}


def compute_ou_zscore(spread, ou_window=252, min_window=2):
    """
    Compute z-score using a rolling window derived from the OU half-life.

    The window is set to the half-life of mean reversion estimated from
    the trailing ou_window bars, so it adapts to the actual reversion speed
    rather than an arbitrary fixed number.

    Parameters
    ----------
    ou_window  : int  — bars used to re-estimate OU params at each step (default 252)
    min_window : int  — lower bound on half-life in bars before a bar is skipped.
                        Default 2; KO/PEP legitimately reverts in ~2 days on log prices.

    Returns
    -------
    zscore    : Series
    half_life : Series  (diagnostic — plot to monitor regime stability)
    """
    zscores    = np.full(len(spread), np.nan)
    half_lives = np.full(len(spread), np.nan)

    spread_vals = spread.values

    for i in range(ou_window, len(spread)):
        ou = fit_ou_params(spread.iloc[i - ou_window:i])

        if ou is None:
            continue

        hl = ou["half_life"]
        if not (min_window <= hl <= ou_window):
            continue  # half-life implausible for this window

        half_lives[i] = hl

        # Z-score window = half-life, floored at 5 bars
        roll_window  = max(20, int(round(hl)))
        window_for_z = spread_vals[i - roll_window + 1: i + 1]
        mu           = window_for_z.mean()
        sigma        = window_for_z.std()

        if sigma <= 0:
            continue

        zscores[i] = (spread_vals[i] - mu) / sigma

    return (
        pd.Series(zscores,    index=spread.index, name="zscore"),
        pd.Series(half_lives, index=spread.index, name="half_life"),
    )


# signal generation

def generate_positions(zscore, entry=2.0, exit_z=0.5, stop=None):
    """
    State-machine signal generator with fixed z-score thresholds.

    Position semantics (long the spread = long PEP, short KO):
        +1 : long spread  (z too negative → expect reversion upward)
        -1 : short spread (z too positive → expect reversion downward)
         0 : flat

    Lookahead fix: position is recorded BEFORE the current bar's z-score
    is acted on, so trades execute at the NEXT bar's open.

    entry  : float — z-score level to enter a position (default 2.0)
    exit_z : float — z-score level to exit a position (default 0.5)
    stop   : float or None — if set, exit when z moves this many units
             beyond the entry threshold. e.g. stop=1.5, entry=2.0
             → long stopped out if z < -3.5
    """
    positions          = []
    pos                = 0
    entry_z_at_open    = np.nan

    for z in zscore:
        positions.append(pos)   # record BEFORE acting — no lookahead

        if pd.isna(z):
            continue

        if pos == 0:
            if z < -entry:
                pos             = 1
                entry_z_at_open = entry
            elif z > entry:
                pos             = -1
                entry_z_at_open = entry

        elif pos == 1:
            if z >= -exit_z:
                pos             = 0
                entry_z_at_open = np.nan
            elif stop is not None and z < -(entry_z_at_open + stop):
                pos             = 0
                entry_z_at_open = np.nan

        elif pos == -1:
            if z <= exit_z:
                pos             = 0
                entry_z_at_open = np.nan
            elif stop is not None and z > (entry_z_at_open + stop):
                pos             = 0
                entry_z_at_open = np.nan

    return pd.Series(positions, index=zscore.index, name="position")


# top-level pipeline

def run_kalman_ou_pipeline(
    log_prices,
    y_col        = "PEP",
    x_col        = "KO",
    kalman_delta = 1e-4,
    kalman_R     = 1e-2,
    ou_window    = 252,
    min_window   = 2,
    entry        = 2.0,
    exit_z       = 0.5,
    stop         = None,
):
    """
    Full pipeline: prices → Kalman hedge ratio → spread → OU z-score → positions.

    Parameters
    ----------
    kalman_delta : float — process noise. Higher = beta adapts faster.
    kalman_R     : float — observation noise.
    ou_window    : int   — bars used to re-estimate OU params each step.
    min_window   : int   — minimum acceptable half-life in bars.
    entry        : float — z-score entry threshold (default 2.0).
    exit_z       : float — z-score exit threshold (default 0.5).
    stop         : float or None — stop-loss in z-score units beyond entry.

    Returns
    -------
    dict with keys:
        kalman_ratios : DataFrame — beta, intercept, uncertainty per bar
        spread        : Series
        zscore        : Series
        half_life     : Series — rolling OU half-life (regime diagnostic)
        positions     : Series
    """
    kalman_ratios = kalman_hedge_ratio(
        log_prices, y_col=y_col, x_col=x_col,
        delta=kalman_delta, R_var=kalman_R,
    )

    spread = compute_kalman_spread(
        log_prices, kalman_ratios, y_col=y_col, x_col=x_col,
    )

    zscore, half_life = compute_ou_zscore(
        spread, ou_window=ou_window, min_window=min_window,
    )

    positions = generate_positions(
        zscore, entry=entry, exit_z=exit_z, stop=stop,
    )

    return {
        "kalman_ratios": kalman_ratios,
        "spread":        spread,
        "zscore":        zscore,
        "half_life":     half_life,
        "positions":     positions,
    }