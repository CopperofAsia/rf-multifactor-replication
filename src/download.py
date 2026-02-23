from __future__ import annotations
import pandas as pd
from pathlib import Path

from src.config import INDEX_CODE, START_DATE, END_DATE, REB_FREQ_TRADE_DAYS, RAW_DIR, INTERIM_DIR
from src.tushare_client import get_pro, safe_call


def get_trade_calendar(pro) -> pd.DataFrame:
    """
    Get trade calendar between START_DATE and END_DATE.
    Return DataFrame with 'cal_date', 'is_open' and filtered open days.
    """
    cal = safe_call(
        pro.trade_cal,
        exchange="SSE",
        start_date=START_DATE,
        end_date=END_DATE,
        fields="cal_date,is_open"
    )
    cal = cal.sort_values("cal_date")
    open_days = cal[cal["is_open"] == 1].copy()
    open_days["cal_date"] = open_days["cal_date"].astype(str)
    return open_days.reset_index(drop=True)


def get_rebalance_dates(open_days: pd.DataFrame, freq: int = REB_FREQ_TRADE_DAYS) -> list[str]:
    """
    Every 'freq' open trading days as rebalance date.
    """
    dates = open_days["cal_date"].tolist()
    # 以第一个交易日为第0期调仓日：0, freq, 2*freq...
    reb_dates = [dates[i] for i in range(0, len(dates), freq)]
    return reb_dates


def download_index_daily(pro, out_path: Path) -> pd.DataFrame:
    """
    Download CSI500 index daily data (index_daily).
    """
    idx = safe_call(
        pro.index_daily,
        ts_code=INDEX_CODE,
        start_date=START_DATE,
        end_date=END_DATE,
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    )
    idx = idx.sort_values("trade_date").reset_index(drop=True)
    idx.to_parquet(out_path, index=False)
    return idx


def download_index_components_at_dates(pro, reb_dates: list[str], out_path: Path) -> pd.DataFrame:
    """
    For each rebalance date, use index_weight to get index constituents (ts_code, weight).
    Note: index_weight uses trade_date. We query exact rebalance date.
    If empty (e.g. not published that day), we fallback to nearest previous rebalance date with data.
    
    注意：
    这个 fallback 机制实际存在问题，如果调仓日没有权重信息，应该取前面的最近的有权重信息的交易日
    的权重信息，但这里为了简化代码，只用前面的最近的有权重信息的调仓日的权重信息。
    """
    records = []
    last_valid_df = None
    last_valid_date = None

    for d in reb_dates:
        df = safe_call(
            pro.index_weight,
            index_code=INDEX_CODE,
            trade_date=d,
            fields="index_code,con_code,trade_date,weight"
        )
        if df.empty:
            # fallback: use last valid constituents
            if last_valid_df is not None:
                tmp = last_valid_df.copy()
                tmp["trade_date"] = d  # 标记本次调仓日沿用上次成分
                records.append(tmp)
            continue
        last_valid_df = df
        last_valid_date = d
        records.append(df)

    if not records:
        raise RuntimeError("index_weight returned empty for all rebalance dates. Check permissions or index code.")

    comp = pd.concat(records, ignore_index=True)
    comp = comp.rename(columns={"con_code": "ts_code"})
    comp = comp.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    comp.to_parquet(out_path, index=False)
    return comp


def _chunk_list(xs: list[str], n: int):
    '''
    Tushare API 有一次最多 200 个代码限制，不能一次请求 1000 只股票，必须分批请求。
    这个函数就是专门为此设计的。
    '''
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def download_stock_daily_panel(pro, ts_codes: list[str], out_basic: Path):
    '''
    Robust downloader:
    - daily_basic: per-ts_code (Tushare often doesn't accept comma-separated ts_code here)
    - daily: try batch first; if batch returns empty, fallback to per-ts_code
    - safe for empty concat; prints clear warnings
    '''
    # all_daily = []
    all_basic = []
    
    '''
    # ---------- 1) daily: try batch ----------
    batch_supported = True
    for batch in _chunk_list(ts_codes, 200):
        code_str = ",".join(batch)
        d1 = safe_call(
            pro.daily,
            ts_code=code_str,
            start_date=START_DATE,
            end_date=END_DATE,
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        )
        if d1 is not None and not d1.empty:
            all_daily.append(d1)
        else:
            # 一旦发现批量完全拿不到，后面直接降级逐个
            batch_supported = False
            break

    # ---------- 2) daily fallback: per code ----------
    if not batch_supported:
        print("[INFO] pro.daily doesn't seem to accept batch ts_code. Falling back to per-ts_code download...")
        all_daily = []
        for code in ts_codes:
            d1 = safe_call(
                pro.daily,
                ts_code=code,
                start_date=START_DATE,
                end_date=END_DATE,
                fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
            )
            if d1 is not None and not d1.empty:
                all_daily.append(d1)
    '''        
    

    # ---------- 3) daily_basic: per code (always) ----------
    for code in ts_codes:
        d2 = safe_call(
            pro.daily_basic,
            ts_code=code,
            start_date=START_DATE,
            end_date=END_DATE,
            fields="ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,"
                   "dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv"
        )
        if d2 is not None and not d2.empty:
            all_basic.append(d2)

    # ---------- 4) concat + save (with empty guards) ----------
    # daily_df = pd.concat(all_daily, ignore_index=True) if len(all_daily) > 0 else pd.DataFrame()
    basic_df = pd.concat(all_basic, ignore_index=True) if len(all_basic) > 0 else pd.DataFrame()

    '''
    if daily_df.empty:
        raise RuntimeError("daily_df is empty. Check tushare daily permissions / params / network.")
    '''
    if basic_df.empty:
        raise RuntimeError("basic_df is empty. daily_basic should work (you verified single-code), "
                           "so likely rate-limit/network. Try smaller universe first.")

    # daily_df = daily_df.drop_duplicates(["ts_code", "trade_date"]).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    basic_df = basic_df.drop_duplicates(["ts_code", "trade_date"]).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # daily_df.to_parquet(out_daily, index=False)
    basic_df.to_parquet(out_basic, index=False)

    return basic_df



def main():
    pro = get_pro()

    # 1) 交易日历 & 调仓日
    open_days = get_trade_calendar(pro)
    (RAW_DIR / "trade_calendar.parquet").write_bytes(open_days.to_parquet(index=False))
    reb_dates = get_rebalance_dates(open_days, REB_FREQ_TRADE_DAYS)
    pd.Series(reb_dates, name="rebalance_date").to_csv(RAW_DIR / "rebalance_dates.csv", index=False)

    # 2) 指数行情
    '''
    idx_path = RAW_DIR / f"index_daily_{INDEX_CODE.replace('.', '_')}.parquet"
    download_index_daily(pro, idx_path)
    '''

    # 3) 每个调仓日的指数成分（关键：严格复现）
    comp_path = RAW_DIR / f"index_components_{INDEX_CODE.replace('.', '_')}.parquet"
    comp = download_index_components_at_dates(pro, reb_dates, comp_path)

    # 4) 汇总全部出现过的成分股代码（为了先把底表拉全；后续回测按调仓日成分过滤）
    universe = sorted(comp["ts_code"].unique().tolist())
    pd.Series(universe, name="ts_code").to_csv(RAW_DIR / "universe_cs500.csv", index=False)

    # 5) 下载全部出现过的股票日行情 + daily_basic
    # out_daily = RAW_DIR / "stock_daily_cs500.parquet"
    out_basic = RAW_DIR / "stock_daily_basic_cs500.parquet"
    download_stock_daily_panel(pro, universe, out_basic)

    print("Step 1 done.")
    # print(f"- index daily: {idx_path}")
    print(f"- components : {comp_path}")
    # print(f"- stock daily: {out_daily}")
    print(f"- daily_basic: {out_basic}")
    print(f"- universe   : {RAW_DIR / 'universe_cs500.csv'}")


if __name__ == "__main__":
    main()
