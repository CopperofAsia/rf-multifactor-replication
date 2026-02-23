import numpy as np
import pandas as pd


def build_price_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily (ts_code, trade_date, close, ...)
    Output: daily with mom_20d, mom_60d
    """
    df = daily.copy()
    df = df.sort_values(["ts_code", "trade_date"])

    # 动量 = P_t / P_{t-k} - 1
    df["mom_20d"] = df.groupby("ts_code")["close"].pct_change(20)
    df["mom_60d"] = df.groupby("ts_code")["close"].pct_change(60)

    return df


def build_basic_factors(basic: pd.DataFrame) -> pd.DataFrame:
    """
    Input: daily_basic
    Output: add log_mktcap, ep, bp, turnover_20d
    """
    df = basic.copy()
    df = df.sort_values(["ts_code", "trade_date"])

    df["log_mktcap"] = np.log(df["total_mv"])
    df["ep"] = 1.0 / df["pe_ttm"]
    df["bp"] = 1.0 / df["pb"]

    df["turnover_20d"] = (
        df.groupby("ts_code")["turnover_rate"]
          .rolling(20)
          .mean()
          .reset_index(level=0, drop=True)
    )

    return df