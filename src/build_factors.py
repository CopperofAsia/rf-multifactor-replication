import pandas as pd
from pathlib import Path

from src.factors import build_price_factors, build_basic_factors
from src.config import RAW_DIR, INTERIM_DIR


def main():
    daily = pd.read_parquet(RAW_DIR / "stock_daily_cs500.parquet")
    basic = pd.read_parquet(RAW_DIR / "stock_daily_basic_cs500.parquet")

    daily_f = build_price_factors(daily)
    basic_f = build_basic_factors(basic)

    # merge 成一个面板表
    '''
    注意：
    how = "left": 左连接，保留 "daily_f" 作为回测时间轴的主索引。
    daily_f 是主表，basic_f 是辅表，以主表的 ["ts_code", "trade_date"] 字段值为准。
    '''
    panel = (
        daily_f.merge(
            basic_f[
                ["ts_code", "trade_date", "log_mktcap", "ep", "bp", "turnover_20d"]
            ],
            on=["ts_code", "trade_date"],
            how="left"
        )
    )

    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    out_path = INTERIM_DIR / "panel_with_factors.parquet"
    panel.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    # print(panel.head())
    # print(panel[["mom_20d", "mom_60d", "ep", "bp", "turnover_20d"]].describe())


if __name__ == "__main__":
    main()