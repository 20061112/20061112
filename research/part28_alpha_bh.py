"""第二十八部分 A：OW 是 alpha 還是 beta？能不能贏買進持有黃金？（M15 OW，2023/3~2026/10）
帳戶：每筆風險 0.25%（不複利，帳戶 % = R × 0.25）；買進持有：同期間黃金現貨（%）。"""
import numpy as np, pandas as pd
from oos_engine import load_bars
from part27_ow import PER

out = open("results_part28.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
T = pd.read_csv("part27_ow_M15.csv", parse_dates=["day", "t_in", "exit_time"])
T = T[T.day >= "2023-03-01"]
idx = [d for d in days if d >= pd.Timestamp("2023-03-01")]
strat = pd.Series(0.0, index=idx).add(T.groupby("day").R.sum() * 0.25, fill_value=0)       # 帳戶 %
gold_cc = (D.close / D.close.shift(1) - 1).reindex(idx) * 100                                # 現貨 日報酬 %（收→收）
gold_oc = ((D.close - D.open) / D.open).reindex(idx) * 100                                   # 當天 開→收 %

def ols(y, x):
    X = np.column_stack([np.ones(len(y)), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; s2 = e @ e / (len(y) - 2); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, b / se, 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))

P("第二十八部分 A：OW 的 alpha / beta 與買進持有比較（策略 = 每筆 0.25% 風險、不複利；黃金 = 現貨 %）")
P(f"  多 / 空 筆數與每筆 R：" + "  ".join(f"{n}: 多 {(x.side == 1).sum()}筆 {x[x.side == 1].R.mean():+.2f}R / 空 {(x.side == -1).sum()}筆 {x[x.side == -1].R.mean():+.2f}R"
                                    for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]]))
for lab, g in [("對黃金 開→收（同一天）", gold_oc), ("對黃金 收→收", gold_cc)]:
    b, t, r2 = ols(strat.values, g.values)
    P(f"  回歸 {lab}：beta {b[1]:+.3f}（t {t[1]:+.2f}），alpha {b[0]:+.3f}%/天（t {t[0]:+.2f}，年化 {b[0] * 250:+.0f}%），R² {r2:.3f}")
up = gold_oc > 0
P(f"  黃金上漲日 {up.sum()} 天 策略日均 {strat[up].mean():+.3f}%；下跌日 {(~up).sum()} 天 策略日均 {strat[~up].mean():+.3f}%")

P("\n  各期比較                        策略（0.25%/筆）          買進持有黃金")
for n, a, b in PER + [("全期", "2023-03-01", "2026-12-31"), ("24H2 起", "2024-07-01", "2026-12-31")]:
    s = strat[(strat.index >= a) & (strat.index <= b)]; g = gold_cc[(gold_cc.index >= a) & (gold_cc.index <= b)].fillna(0)
    se, ge = s.cumsum(), (1 + g / 100).cumprod()
    sh = lambda x: x.mean() / x.std() * np.sqrt(250) if x.std() > 0 else np.nan
    P(f"  {n:8s} 策略 {se.iloc[-1]:+6.1f}% 回撤 {(se - se.cummax()).min():6.1f}% Sharpe {sh(s):5.2f} | "
      f"黃金 {(ge.iloc[-1] - 1) * 100:+6.1f}% 回撤 {((ge / ge.cummax()) - 1).min() * 100:6.1f}% Sharpe {sh(g):5.2f} | 相關 {s.corr(g):+.2f}")
# 槓桿到與黃金同波動
s = strat[strat.index >= "2024-07-01"]; g = gold_cc[gold_cc.index >= "2024-07-01"].fillna(0)
k = g.std() / s.std()
P(f"\n  2024H2 起：把策略風險放大到與黃金同樣的日波動（約每筆 {0.25 * k:.2f}% 風險）→ 策略 {s.sum() * k:+.0f}% vs 黃金 {((1 + g / 100).prod() - 1) * 100:+.0f}%"
  f"（同期間 Sharpe 策略 {s.mean() / s.std() * np.sqrt(250):.2f} / 黃金 {g.mean() / g.std() * np.sqrt(250):.2f}）")
P(f"  兩者相關 {s.corr(g):+.2f} → 一半資金抱黃金 + 一半跑策略（同波動）：Sharpe "
  f"{((s * k + g) / 2).mean() / ((s * k + g) / 2).std() * np.sqrt(250):.2f}")
out.close()
