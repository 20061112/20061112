"""檢查第三十四部分的浮動回撤：找出高點 / 低點時刻，拆解成已實現 + 未實現，並用不同價格口徑重算。"""
import numpy as np, pandas as pd
from part27_ow import data

T = pd.read_csv("part31_m1_trades.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
days, D, ARR = data("data/XAUUSD_M1_2026.csv", 1)


def build(T, price="worst"):
    rows = []
    for n, r in enumerate(T.itertuples()):
        t, O, H, L, C, S, mins = ARR[r.day]
        j = int(np.searchsorted(t, r.exit_time - pd.Timedelta(minutes=1)))
        if j > r.i:
            px = {"worst": (L if r.side == 1 else H), "close": C, "best": (H if r.side == 1 else L)}[price][r.i:j]
            rows.append(pd.Series(r.side * (px - r.lvl), index=t[r.i:j], name=n))
    U = pd.concat(rows, axis=1).sort_index()
    op = U.sum(axis=1, min_count=1).fillna(0)
    real = T.groupby("exit_time").pnl.sum().sort_index()
    ix = op.index.union(real.index)
    R = real.reindex(ix, fill_value=0).cumsum(); O_ = op.reindex(ix, fill_value=0)
    return R + O_, R, O_, U


eq, R, O_, U = build(T, "worst")
dd = eq - eq.cummax(); tr = dd.idxmin(); pk = eq[:tr].idxmax()
print(f"含浮動（K 棒最差價）最大回撤 {dd.min():.0f}：高點 {pk} 權益 {eq[pk]:.0f}（已實現 {R[pk]:.0f} + 未實現 {O_[pk]:.0f}）"
      f" → 低點 {tr} 權益 {eq[tr]:.0f}（已實現 {R[tr]:.0f} + 未實現 {O_[tr]:.0f}）")
print(f"  高點時持倉 {int(U.loc[pk].notna().sum()) if pk in U.index else 0} 筆；低點時持倉 {int(U.loc[tr].notna().sum()) if tr in U.index else 0} 筆")
closed = T.sort_values("exit_time").pnl.cumsum()
print(f"平倉後最大回撤 {(closed - closed.cummax()).min():.0f}")
for p in ("close", "worst"):
    e = build(T, p)[0]; print(f"  用 {p} 價算未實現：最大回撤 {(e - e.cummax()).min():.0f}")
# 檢查：期末權益應 = 總損益
print(f"期末權益 {eq.iloc[-1]:.0f} vs 總損益 {T.pnl.sum():.0f}（差 = 期末還開著的單，應為 0）")
# 高點那天的明細
d = pk.normalize() if pk.hour >= 1 else pk.normalize() - pd.Timedelta(days=1)
x = T[(T.t_in <= pk) & (T.exit_time > pk)]
print(f"\n高點時刻開著的單（{len(x)} 筆）：")
print(x[["t_in", "side", "lvl", "ibh", "ibl", "risk", "exit_time", "exit_reason", "pnl"]].to_string(index=False))
y = T[(T.t_in <= tr) & (T.exit_time > tr)]
print(f"\n低點時刻開著的單（{len(y)} 筆）：")
print(y[["t_in", "side", "lvl", "risk", "exit_time", "exit_reason", "pnl"]].to_string(index=False))
print(f"\n高點到低點之間平倉的單：{len(T[(T.exit_time > pk) & (T.exit_time <= tr)])} 筆，合計 {T[(T.exit_time > pk) & (T.exit_time <= tr)].pnl.sum():.0f}")

# 另一種口徑：只從「已實現（平倉）高點」往下算，浮動獲利的回吐不算回撤
realpk = R.cummax()
print(f"\n從平倉高點起算的最大浮動回撤（不把未實現獲利當高點）：{(eq - realpk).min():.0f}，發生在 {(eq - realpk).idxmin()}")
# 3/23 這天：加 3R/2R 移動停損會怎樣
from part32_constraints import manage
d23 = T[T.day == "2026-03-23"]
tr_r = [manage(r, ARR, 1, trail=(3, 2))[0] for r in d23.itertuples()]
print(f"3/23 共 {len(d23)} 筆：現行合計 {d23.pnl.sum():+.0f} 美元；加 3R/2R 移動停損 → {sum(x * r.risk for x, r in zip(tr_r, d23.itertuples())):+.0f} 美元")
t, O, H, L, C, S, mins = ARR[pd.Timestamp("2026-03-23")]
print(f"3/23 金價：開盤 {O[0]:.0f}，最低 {L.min():.0f}（{t[L.argmin()]}），之後最高 {H[L.argmin():].max():.0f}，收盤 {C[-1]:.0f}")
