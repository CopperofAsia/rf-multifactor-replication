import os
import time
from typing import Optional

import tushare as ts
import pandas as pd


def get_pro(token: Optional[str] = None):
    """
    Create tushare pro client.
    Recommend: set env var TUSHARE_TOKEN.
    """
    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError(
            "Tushare token not found. Please set env var TUSHARE_TOKEN "
            "or pass token explicitly."
        )
    ts.set_token(token)
    return ts.pro_api()


def safe_call(func, max_retry: int = 5, sleep_sec: float = 0.8, **kwargs) -> pd.DataFrame:
    """
    Basic retry wrapper to survive transient failures / rate-limits.
    """
    last_err = None
    for i in range(max_retry):
        try:
            df = func(**kwargs)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec * (i + 1))
    raise last_err