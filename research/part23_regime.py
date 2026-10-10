"""第二十三部分 C：市場層級的狀態（全部用「今天以前」的資料、無單位）能否預測今天的勝率。
  eff20   過去 20 天日內效率平均（|收−開|/區間）
  rev20   過去 20 天紐約盤反轉比例
  pers20  過去 20 天「今天方向 = 昨天方向」的比例（日線趨勢延續性）
  win50   策略最近 50 筆已平倉交易的勝率（策略自身的狀態）
  atr_ch  ATR10 / ATR60（波動擴張或收縮，無單位）
各特徵切三等分（用全期分位），看每年各組的勝率與每筆 R。"""
import numpy as np, pandas as pd
from oos_engine import load_bars

out = open("results_part23.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
X = pd.read_csv("part23_trades_states.csv", parse_dates=["day", "t_in", "exit_time"])
R = pd.read_csv("oos_regime_days.csv", parse_dates=["day"]).set_index("day")
days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
D["dir"] = np.sign(D.close - D.open)
D["eff"] = (D.close - D.open).abs() / (D.high - D.low)
D["pers"] = (D.dir == D.dir.shift(1)).astype(float)
D["atr60"] = D.tr.rolling(60).mean().shift(1)
rev = R.rev.reindex(D.index)
F = pd.DataFrame({"eff20": D.eff.rolling(20).mean().shift(1), "rev20": rev.rolling(20, min_periods=10).mean().shift(1),
                  "pers20": D.pers.rolling(20).mean().shift(1), "atr_ch": D.atr10 / D.atr60})
X = X.join(F, on="day")
# 策略最近 50 筆已平倉勝率（以當天開盤前已平倉的交易）
Xs = X.sort_values("exit_time"); wins = []
ex_t, ex_w = Xs.exit_time.values, Xs.win.values
for d in X.day:
    k = np.searchsorted(ex_t, np.datetime64(d)); w = ex_w[max(0, k - 50):k]
    wins.append(w.mean() if len(w) >= 30 else np.nan)
X["win50"] = wins
X["yr"] = np.where(X.day < "2025-07-01", X.day.dt.year.astype(str), np.where(X.day < "2026-01-01", "25H2", "2026"))
X.loc[(X.day >= "2025-01-01") & (X.day < "2025-07-01"), "yr"] = "25H1"
pf = lambda p: p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)
P("\n" + "=" * 100 + "\n第二十三部分 C：市場層級狀態（事前可知）→ 勝率 / 每筆R（三等分：低 / 中 / 高）")
for f in ["eff20", "rev20", "pers20", "win50", "atr_ch"]:
    q = pd.qcut(X[f], 3, labels=["低", "中", "高"])
    P(f"\n  {f}（分界 {X[f].quantile(1/3):.2f} / {X[f].quantile(2/3):.2f}）")
    for lv in ["低", "中", "高"]:
        cells = []
        for y in ["2023", "2024", "25H1", "25H2", "2026"]:
            x = X[(q == lv) & (X.yr == y)]
            cells.append(f"{y} {x.win.mean() * 100:3.0f}%/{x.R.mean():+.2f}/PF{pf(x.pnl):.2f}({len(x)})")
        P(f"    {lv}  " + "  ".join(cells))
X.to_csv("part23_trades_regime.csv", index=False)
out.close()
