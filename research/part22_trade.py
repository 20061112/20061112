"""把最有希望的組合變成可交易規則，用 M1 逐根模擬（停損 / session 結束出場），以 R 計。

進場：OR 突破那根 M1 收盤價（part22_breakout.csv 的事件）。
停損：OR 中點（stop=mid）或 OR 另一側（stop=opp）。出場：停損或 session 結束。每筆扣一次點差。
規則：
  BASE  所有真 OR 突破
  F1    突破時價格已在前日價值區外（AMT initiative：離開價值、被接受的方向）
  F1x   F1 但 VA 內開盤或 VA 外開盤往遠離價值方向（排除往回價值方向）
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import load_m1, arrays, tstat, SESS_ORDER

d = load_m1()
H, L, C, O, SP = arrays(d)
_, S = pickle.load(open("part22_sessions.pkl", "rb"))
S["day"] = S.day.astype(str)
B = pd.read_csv("part22_breakout.csv")
B = B[B.kind == "real"].merge(S[["day", "sess", "orh", "orl", "orm", "i_or", "i1"]], on=["day", "sess"])
idx = pd.Series(np.arange(len(d)), index=d.index)


def sim(r, stop_kind):
    dr = r.dir
    j = r.i_or + int(r.delay)
    px = C[j]
    stop = r.orm if stop_kind == "mid" else (r.orl if dr == 1 else r.orh)
    risk = (px - stop) * dr
    if risk <= 0:
        return np.nan, np.nan
    seg_l, seg_h = L[j + 1:r.i1 + 1], H[j + 1:r.i1 + 1]
    hit = np.where(seg_l <= stop)[0] if dr == 1 else np.where(seg_h >= stop)[0]
    exit_px = stop if len(hit) else C[r.i1]
    pnl = (exit_px - px) * dr - SP[j]
    return pnl / risk, pnl


rules = {"BASE": B, "F1": B[B.beyond_va], "F1x": B[B.beyond_va & (B.rel != "toward")]}
out = ["=" * 110, "G. 交易版本（M1 逐根，停損或 session 結束出場，扣點差；R = 進場到停損距離）", "=" * 110]
for stop_kind in ("mid", "opp"):
    out.append(f"\n  停損 = {'OR 中點' if stop_kind == 'mid' else 'OR 另一側'}")
    for name, g in rules.items():
        for s in SESS_ORDER + ["ALL"]:
            gg = g if s == "ALL" else g[g.sess == s]
            res = np.array([sim(r, stop_kind) for r in gg.itertuples()])
            R, usd = res[:, 0], res[:, 1]
            ok = ~np.isnan(R)
            R, usd, yrs = R[ok], usd[ok], gg.year.values[ok]
            pf = usd[usd > 0].sum() / -usd[usd < 0].sum()
            yr = " ".join(f"{R[yrs == y].mean():+.2f}" for y in (2023, 2024, 2025, 2026))
            a, b = R[yrs <= 2024], R[yrs >= 2025]
            out.append(f"    {name:<5}{s:<5} n={len(R):5d}  勝率 {(R > 0).mean()*100:4.1f}%  每筆 {R.mean():+.3f}R (t {tstat(R):+4.1f})  "
                       f"PF {pf:4.2f}  發現 {a.mean():+.3f} / 驗證 {b.mean():+.3f}R  逐年 {yr}")
txt = "\n".join(out)
print(txt)
open("results_part22_trade.txt", "w").write(txt + "\n")
