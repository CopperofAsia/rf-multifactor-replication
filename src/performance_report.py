import numpy as np
import pandas as pd

REB_FREQ_TRADE_DAYS = 20  # 与回测一致


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return dd.min()


def perf_stats(period_ret: pd.Series, nav: pd.Series) -> dict:
    periods_per_year = 252 / REB_FREQ_TRADE_DAYS

    total_ret = nav.iloc[-1] - 1.0
    ann_ret = (nav.iloc[-1]) ** (periods_per_year / len(period_ret)) - 1.0

    vol = period_ret.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (period_ret.mean() / period_ret.std(ddof=1)) * np.sqrt(periods_per_year)

    mdd = abs(max_drawdown(nav))
    calmar = ann_ret / mdd if mdd > 0 else np.nan

    return {
        "Total Return": total_ret,
        "Annualized Return": ann_ret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Calmar Ratio": calmar
    }


def main():
    df = pd.read_parquet("outputs/backtest_result.parquet")

    strat_stats = perf_stats(df["port_ret"], df["nav"])
    bench_stats = perf_stats(df["bench_ret"], df["bench_nav"])

    table = pd.DataFrame(
        [strat_stats, bench_stats],
        index=["Random Forest Strategy", "CSI 500 Benchmark"]
    )

    print("\n=== Performance Summary (20-trade-day period) ===")
    print(table)

    out_path = "outputs/performance_table.csv"
    table.to_csv(out_path)
    print("\nSaved performance table to:", out_path)


if __name__ == "__main__":
    main()
