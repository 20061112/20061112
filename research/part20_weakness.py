"""第二十部分：版本 A 的弱點 —— 虧損日 / 虧損週有什麼共同點。

每個交易日（不含週一）計算：
  事後才知道的（描述「怎麼輸的」）：
    rng_atr   當天波幅 / ATR10
    eff       效率 = |收 − 開| / 波幅（1 = 單邊、0 = 來回）
    rev_ny    紐約盤反轉：11:00 時的方向（相對開盤）與 15:30→收盤 的方向相反
    ny_move   15:30 之後的波幅 / ATR10
    mixed     當天同時有多單和空單
  事前就知道的（可能拿來當篩選）：
    atr_ratio ATR10 / ATR60（波動上升或下降中）
    atr_lvl   ATR10 相對過去 60 天的分位
    prev_eff  前一天效率；prev_rng 前一天波幅 / ATR
    gap       開盤跳空 / ATR
    asia_rng  01:00~10:00 的波幅 / ATR（進場前就知道一部分）
"""
import numpy as np, pandas as pd
from part14_deep import ARR, D, DAYS
from part17_balance import gen
from final_A import CFG, walk_forward

out = open("results_part20.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

T, _ = walk_forward(gen(**CFG))
D["rng"] = D.high - D.low
D["atr60"] = D.tr.rolling(60, min_periods=20).mean().shift(1)
D["eff"] = (D.close - D.open).abs() / D.rng


def at(d, hh, mm=0):
    t, O, H, L, C, S = ARR[d]; k = np.searchsorted(t, d + pd.Timedelta(hours=hh, minutes=mm))
    return min(k, len(C) - 1)


rows = []
for d in DAYS:
    if d < T.day.min() or d > T.day.max() or d.dayofweek == 0:
        continue
    di = DAYS.index(d); t, O, H, L, C, S = ARR[d]; atr = D.atr10[d]; prev = D.iloc[di - 1]
    k11, k1530 = at(d, 11), at(d, 15, 30)
    dir_am = np.sign(C[k11] - O[0]); dir_ny = np.sign(C[-1] - C[k1530])
    x = T[T.day == d]
    hist = D.atr10.iloc[max(0, di - 60):di]
    rows.append(dict(day=d, pnl=x.pnl.sum(), n=len(x), n_long=(x.side == 1).sum(), n_short=(x.side == -1).sum(),
                     stops=(x.exit_reason == "停損").sum(),
                     rng_atr=(H.max() - L.min()) / atr, eff=abs(C[-1] - O[0]) / (H.max() - L.min()),
                     rev_ny=int(dir_am != 0 and dir_ny == -dir_am), ny_move=(H[k1530:].max() - L[k1530:].min()) / atr,
                     am_move=abs(C[k11] - O[0]) / atr, mixed=int((x.side == 1).any() and (x.side == -1).any()),
                     atr=atr, atr_ratio=atr / D.atr60[d], atr_lvl=(hist < atr).mean() if len(hist) > 10 else np.nan,
                     prev_eff=prev.eff, prev_rng=prev.rng / atr, gap=abs(O[0] - prev.close) / atr,
                     asia_rng=(H[:at(d, 10)].max() - L[:at(d, 10)].min()) / atr, wd=d.dayofweek))
G = pd.DataFrame(rows).set_index("day")
G["state"] = np.where(G.n == 0, "沒交易", np.where(G.pnl > 0, "獲利日", "虧損日"))

P("第二十部分：版本 A 的弱點（滾動前推 3~10 月）")
P(f"  交易日 {len(G)}：獲利日 {(G.state == '獲利日').sum()}、虧損日 {(G.state == '虧損日').sum()}、沒交易 {(G.state == '沒交易').sum()}")
L_, W_ = G[G.state == "虧損日"], G[G.state == "獲利日"]
P(f"  虧損日平均 {L_.pnl.mean():+.1f}（最差 {L_.pnl.min():+.0f}），獲利日平均 {W_.pnl.mean():+.1f}")

P("\n1) 虧損日 vs 獲利日：特徵平均（中位數）")
feats = [("rng_atr", "當天波幅/ATR"), ("eff", "效率 |收−開|/波幅"), ("rev_ny", "紐約盤反轉比例"), ("ny_move", "15:30後波幅/ATR"),
         ("am_move", "11:00 前走了多少/ATR"), ("mixed", "多空都有開"), ("n", "交易筆數"), ("stops", "停損筆數"),
         ("atr", "ATR10"), ("atr_ratio", "ATR10/ATR60"), ("atr_lvl", "ATR 在過去 60 天的分位"), ("prev_eff", "前一天效率"),
         ("prev_rng", "前一天波幅/ATR"), ("gap", "開盤跳空/ATR"), ("asia_rng", "01~10 點波幅/ATR")]
for f, lab in feats:
    P(f"  {lab:22s} 虧損日 {L_[f].mean():6.2f}（{L_[f].median():6.2f}）  獲利日 {W_[f].mean():6.2f}（{W_[f].median():6.2f}）")

P("\n2) 依特徵分組的日均損益與虧損日比例")
for f, lab in feats:
    if f in ("n", "stops", "mixed", "rev_ny"):
        g = G[f].clip(upper=G[f].quantile(.9)) if f in ("n", "stops") else G[f]
        grp = pd.cut(g, [-1, 2, 5, 8, 99], labels=["0~2", "3~5", "6~8", "9+"]) if f in ("n", "stops") else g
    else:
        grp = pd.qcut(G[f], 4, labels=["低 25%", "25~50%", "50~75%", "高 25%"], duplicates="drop")
    a = G.groupby(grp, observed=True).agg(天=("pnl", "size"), 日均=("pnl", "mean"), 虧損日=("state", lambda s: (s == "虧損日").mean() * 100))
    P(f"  {lab:22s} " + "  ".join(f"[{k}] {int(v['天'])}天 {v['日均']:+6.1f} 虧{v['虧損日']:.0f}%" for k, v in a.iterrows()))

P("\n3) 最差 15 天")
P(G.sort_values("pnl").head(15)[["pnl", "n", "n_long", "n_short", "stops", "rng_atr", "eff", "rev_ny", "am_move", "ny_move", "atr_ratio", "prev_eff"]]
  .round(2).to_string())

P("\n4) 週")
G["week"] = G.index.to_period("W")
Wk = G.groupby("week").agg(pnl=("pnl", "sum"), days=("pnl", "size"), lose_days=("state", lambda s: (s == "虧損日").sum()),
                           rng=("rng_atr", "mean"), eff=("eff", "mean"), rev=("rev_ny", "mean"), atr_ratio=("atr_ratio", "mean"))
P(f"  共 {len(Wk)} 週，虧損週 {(Wk.pnl < 0).sum()}（{(Wk.pnl < 0).mean() * 100:.0f}%）")
P(f"  虧損週 平均：日波幅/ATR {Wk[Wk.pnl < 0].rng.mean():.2f}、效率 {Wk[Wk.pnl < 0].eff.mean():.2f}、紐約反轉 {Wk[Wk.pnl < 0].rev.mean():.2f}、ATR10/60 {Wk[Wk.pnl < 0].atr_ratio.mean():.2f}")
P(f"  獲利週 平均：日波幅/ATR {Wk[Wk.pnl > 0].rng.mean():.2f}、效率 {Wk[Wk.pnl > 0].eff.mean():.2f}、紐約反轉 {Wk[Wk.pnl > 0].rev.mean():.2f}、ATR10/60 {Wk[Wk.pnl > 0].atr_ratio.mean():.2f}")
P(Wk[Wk.pnl < 0].round(2).to_string())

P("\n5) 虧損集中在哪：停損出場時間（虧損日）")
loss_tr = T[T.day.isin(L_.index) & (T.exit_reason == "停損")]
P("  " + "  ".join(f"{h:02d}時 {c}" for h, c in loss_tr.exit_time.dt.hour.value_counts().sort_index().items()))
P(f"  虧損日的停損中，發生在 15:30 之後（紐約盤）的比例 {(loss_tr.exit_time.dt.hour * 60 + loss_tr.exit_time.dt.minute >= 930).mean() * 100:.0f}%；"
  f"獲利日 {(T[T.day.isin(W_.index) & (T.exit_reason == '停損')].exit_time.dt.hour * 60 + T[T.day.isin(W_.index) & (T.exit_reason == '停損')].exit_time.dt.minute >= 930).mean() * 100:.0f}%")
G.to_csv("part20_days.csv")
out.close()
