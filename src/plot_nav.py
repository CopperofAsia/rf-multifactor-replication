import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 读取数据
df = pd.read_parquet("outputs/backtest_result.parquet")
df.index = pd.to_datetime(df.index)

save_path = Path("outputs/nav_comparison.png")

plt.figure(figsize=(10, 5))

plt.plot(df.index, df["nav"], label="Random Forest Strategy")
plt.plot(df.index, df["bench_nav"], label="CSI 500 Benchmark")

plt.title("Net Asset Value Comparison")
plt.xlabel("Rebalance Date")
plt.ylabel("Net Asset Value")
plt.legend()
plt.grid(True)

# 自动控制时间刻度
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())        # 每年一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # 显示年份

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()