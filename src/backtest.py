import joblib
import numpy as np
import pandas as pd

from src.config import RAW_DIR, INTERIM_DIR, MODEL_DIR, OUTPUT_DIR, INDEX_CODE, REB_FREQ_TRADE_DAYS
from src.config import BT_START, BT_END
from src.preprocess import preprocess_factors


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()  # 计算“到当前时刻为止净值的历史最高值”
    dd = nav / peak - 1.0
    return dd.min()


def perf_stats(period_ret: pd.Series, nav: pd.Series) -> dict:
    """
    输入“每期收益序列”（每期20个交易日）和净值曲线 nav，输出一组常用绩效指标：总收益、年化收益、
    年化波动、夏普比、最大回撤、卡玛比
    Args:
        period_ret: 每期(20日)收益，维数：T*1，T = 成功生成回测的调仓期数量
        nav：从 1 开始的累计净值，维数：与 period_ret 相同，nav_t = \prod_{i=1}^t period_ret_i
        
    Return:
        (dict): 常用绩效指标
    """
    periods_per_year = 252 / REB_FREQ_TRADE_DAYS
    total_ret = nav.iloc[-1] - 1.0
    ann_ret = (nav.iloc[-1]) ** (periods_per_year / len(period_ret)) - 1.0 if len(period_ret) > 0 else np.nan
    vol = period_ret.std(ddof=1) * np.sqrt(periods_per_year) if len(period_ret) > 1 else np.nan
    sharpe = (period_ret.mean() / period_ret.std(ddof=1)) * np.sqrt(periods_per_year) if len(period_ret) > 1 else np.nan
    mdd = abs(max_drawdown(nav))
    calmar = ann_ret / mdd if (mdd is not None and mdd > 0) else np.nan
    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar
    }


def main():
    # 读取模型
    pack = joblib.load(MODEL_DIR / "rf_model_final.joblib")
    model = pack["model"]
    factor_cols = pack["factor_cols"]

    # 读取数据
    panel = pd.read_parquet(INTERIM_DIR / "panel_with_labels.parquet")
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # 指数行情（用于基准）
    idx = pd.read_parquet(RAW_DIR / f"index_daily_{INDEX_CODE.replace('.', '_')}.parquet")
    idx = idx.sort_values("trade_date").reset_index(drop=True)

    # 调仓日
    reb = pd.read_csv(RAW_DIR / "rebalance_dates.csv")["rebalance_date"].astype(str).tolist()

    # 指数成分（严格按当期成分）
    comp = pd.read_parquet(RAW_DIR / f"index_components_{INDEX_CODE.replace('.', '_')}.parquet")
    comp = comp.copy()

    # 回测区间
    reb = [d for d in reb if (d >= BT_START and d <= BT_END)]
    if len(reb) == 0:
        # 若smoke test不覆盖研报区间，退化为“用后20%调仓日”
        all_reb = pd.read_csv(RAW_DIR / "rebalance_dates.csv")["rebalance_date"].astype(str).tolist()
        reb = all_reb[int(len(all_reb) * 0.8):]
        print("[INFO] Backtest fallback: using last 20% rebalance dates in sample.")

    # 初始化回测序列容器
    period_rets = []  # 策略每期收益
    bench_rets = []  # 基准指数每期收益
    dates_used = []  # 真正成功生成一期回测的调仓日

    # 回测主循环：对每个调仓日 d 做一次“打分选股 -> 算收益”
    for d in reb:
        # 取当期成分股（若该日无成分记录，跳过）
        comp_d = comp[comp["trade_date"] == d]
        if comp_d.empty:
            continue
        universe = set(comp_d["ts_code"].astype(str).tolist())

        # 取该调仓日 d 的股票横截面快照（snapshot）
        snap = panel[(panel["trade_date"] == d) & (panel["ts_code"].isin(universe))].copy()
        if snap.empty:
            continue

        # 需要未来20日收益用于“这一期”的实际收益
        snap = snap.dropna(subset=["ret_fwd_20d"])
        if snap.empty:
            continue

        # 横截面预处理（只用当日横截面）
        snap = preprocess_factors(snap, factor_cols)

        # 用模型打分（以预测为1的概率作为指标）
        X = snap[factor_cols].values
        prob = model.predict_proba(X)[:, 1]
        snap["prob"] = prob

        # 选前10只
        top = snap.sort_values("prob", ascending=False).head(10)

        # 组合当期收益：等权持有20天（用 ret_fwd_20d 简化）
        port_ret = top["ret_fwd_20d"].mean()

        # 基准收益：指数未来20日收益
        idx_row = idx[idx["trade_date"] == d]
        if idx_row.empty:
            continue
        idx_close = float(idx_row["close"].iloc[0])
        
        # 计算指数未来20日收益
        pos = idx.index[idx["trade_date"] == d][0]  #找到“调仓日 d”在指数表里的位置索引
        if pos + REB_FREQ_TRADE_DAYS >= len(idx):
            continue
        idx_ret = idx.loc[pos + REB_FREQ_TRADE_DAYS, "close"] / idx_close - 1.0

        # 更新回测序列容器
        period_rets.append(port_ret)
        bench_rets.append(idx_ret)
        dates_used.append(d)

    if len(period_rets) == 0:
        raise RuntimeError("No backtest periods generated. Check rebalance dates / data coverage.")

    # 把 list 变成带索引的 Series
    period_rets = pd.Series(period_rets, index=pd.Index(dates_used, name="rebalance_date"), name="port_ret")
    bench_rets = pd.Series(bench_rets, index=pd.Index(dates_used, name="rebalance_date"), name="bench_ret")

    nav = (1.0 + period_rets).cumprod()
    bench_nav = (1.0 + bench_rets).cumprod()

    s_port = perf_stats(period_rets, nav)
    s_bench = perf_stats(bench_rets, bench_nav)

    print("\n=== Backtest Summary (period = 20 trade days) ===")
    print("Strategy:", s_port)
    print("Benchmark:", s_bench)

    out = pd.DataFrame({
        "port_ret": period_rets,
        "bench_ret": bench_rets,
        "nav": nav,
        "bench_nav": bench_nav
    })
    out_path = OUTPUT_DIR / "backtest_result.parquet"
    out.to_parquet(out_path, index=True)
    print("\nSaved backtest series:", out_path)


if __name__ == "__main__":
    main()
