import numpy as np
import pandas as pd


def winsorize_series(s, lower=0.01, upper=0.99):
    """
    对一个 pandas Series 按给定分位数进行去极值（Winsorize）：
    把极端小的值压到下分位，把极端大的值压到上分位，其余值保持不变。
    注意：
    1. s.clip() 是 pandas.Series 的一个方法，用来对数据进行截断（clipping），
    把小于 lower 的值变成 lower，把大于 upper 的值变成 upper。
    Args:
        s (pd.Series): 一只因子在一个截面上的取值
        lower：下分位数，默认 1%
        upper：上分位数，默认 99%
        
    Returns:
        pd.Series    
    """
    if s.notna().sum() == 0:   # s 为空
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def zscore_series(s):
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return s - mu
    return (s - mu) / sigma


def preprocess_factors(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """
    Cross-sectional preprocessing by trade_date:
    1. winsorize
    2. fillna (median)
    3. z-score
    Args:
        df (pd.DataFrame): 多因子 panel 数据
        factor_cols：需要处理的因子列名列表
        
    Returns:
        pd.DataFrame
    """
    out = df.copy()  # 不修改原始 df

    def _process_one_day(x):
        # x: 某一个交易日下的横截面 DataFrame
        x = x.copy()
        for col in factor_cols:
            x[col] = winsorize_series(x[col])
            x[col] = x[col].fillna(x[col].median())
            x[col] = zscore_series(x[col])
        return x

    out = out.groupby("trade_date", group_keys=False).apply(_process_one_day)
    return out
