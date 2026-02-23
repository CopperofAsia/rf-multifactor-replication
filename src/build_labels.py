import pandas as pd
import numpy as np

from src.config import RAW_DIR, INTERIM_DIR, INDEX_CODE, REB_FREQ_TRADE_DAYS


def main():
    '''
    研报定义的标签：下一期个股相对基准的超额收益。
    分类方式：
    Top 30% -> 正样本 -> 1
    Bottom 30% -> 负样本 -> 0
    中间 40% -> 丢弃 -> np.nan
    注意：
    1. .shift(n) 是将某一列整体下移 n 行，而这里用的是 .shift(-n) ，从而可以计算未来的简单收益。
    2. np.where(condition, A, B): A -> 条件为真时取的值；B -> 条件为假时取的值。
    '''
    panel = pd.read_parquet(INTERIM_DIR / "panel_with_factors.parquet")
    idx = pd.read_parquet(RAW_DIR / f"index_daily_{INDEX_CODE.replace('.', '_')}.parquet")

    panel = panel.sort_values(["ts_code", "trade_date"])
    idx = idx.sort_values("trade_date")

    # === 1. 计算未来 20 日收益 ===
    panel["ret_fwd_20d"] = (
        panel.groupby("ts_code")["close"]
        .shift(-REB_FREQ_TRADE_DAYS) / panel["close"] - 1
    )

    idx["ret_fwd_20d"] = idx["close"].shift(-REB_FREQ_TRADE_DAYS) / idx["close"] - 1

    # === 2. merge 指数收益 ===
    panel = panel.merge(
        idx[["trade_date", "ret_fwd_20d"]].rename(columns={"ret_fwd_20d": "ret_idx_20d"}),
        on="trade_date",
        how="left"
    )

    # === 3. 超额收益 ===
    panel["excess_ret_20d"] = panel["ret_fwd_20d"] - panel["ret_idx_20d"]

    # === 4. 构造分类标签（Top30 / Bottom30） ===
    def label_by_date(x):
        q_low = x.quantile(0.3)
        q_high = x.quantile(0.7)
        return np.where(
            x >= q_high, 1,
            np.where(x <= q_low, 0, np.nan)
        )

    panel["label"] = (
        panel.groupby("trade_date")["excess_ret_20d"]
        .transform(label_by_date)
    )

    out = INTERIM_DIR / "panel_with_labels.parquet"
    panel.to_parquet(out, index=False)

    print("Saved:", out)
    # print(panel[["excess_ret_20d", "label"]].describe())
    # print(panel["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
