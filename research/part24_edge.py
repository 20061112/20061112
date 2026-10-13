"""第二十四部分：箱體邊緣反向（區間均值回歸）。輸出 results_part24.txt、part24_grid.csv。

第二十二部分的安慰劑發現：任意 L 根箱體，現價在箱底 20% 內時，先收盤突破箱頂的機率比隨機漫步高約 13 個百分點
（箱頂附近則相反）。這裡把它做成交易，看扣掉點差、用實際停損後還剩多少。

訊號（K 棒 t 收盤時，箱體 = 包含 t 的最近 L 根最高 / 最低，與安慰劑相同）：
  pos = (close − 箱底) / 箱高；pos ≤ zone → 做多；pos ≥ 1 − zone → 做空；箱高 ≥ 0.5 ATR
  過濾（可選）：ER(4L) < 0.3（較長的一段沒有明顯趨勢）
停損：
  close  = 收盤跌破箱底（做多）才在該收盤出場（等同安慰劑的定義）
  touch0 / touch25 / touch50 = 盤中碰到 箱底 − 0 / 0.25 / 0.5 ATR
停利：箱體中點 / 箱頂 / 1R / 2R（close 停損的 R 以 箱底 − 0.25 ATR 計）；最長 H 根收盤出場
同一時間只持有一筆；同一根 K 棒同時碰停損與停利 → 停損；每筆扣進場點差。
研究期 2023~2024 選參數，驗證期 2025~2026/10 只看一次。
"""
import itertools
import numpy as np
import pandas as pd
from load import load
from backtest import resample
from part22_flag import _atr, summ

CUT = pd.Timestamp("2025-01-01")
TFS = {"M5": dict(atr_n=288, H=48), "M15": dict(atr_n=96, H=32), "H1": dict(atr_n=120, H=24)}
LS, ZONES, STOPS, TGTS, FILTS = (5, 10, 20), (0.1, 0.2), ("close", "touch0", "touch25", "touch50"), \
    ("中點", "對邊", "1R", "2R"), ("無", "ER低")


def prep(df, atr_n):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    return dict(h=h, l=l, c=c, sp=df.spread.to_numpy(float) * 0.01, atr=_atr(h, l, c, atr_n),
                idx=df.index, cs=pd.Series(c))


def signals(P, L, zone, filt):
    h, l, c, atr = P["h"], P["l"], P["c"], P["atr"]
    hi = pd.Series(h).rolling(L).max().to_numpy()
    lo = pd.Series(l).rolling(L).min().to_numpy()
    box = hi - lo
    pos = (c - lo) / np.where(box > 0, box, np.nan)
    ok = (box >= 0.5 * atr) & np.isfinite(pos)
    if filt == "ER低":
        n = 4 * L
        er = (P["cs"].diff(n).abs() / P["cs"].diff().abs().rolling(n).sum()).to_numpy()
        ok &= er < 0.3
    d = np.where(ok & (pos <= zone), 1, np.where(ok & (pos >= 1 - zone), -1, 0))
    t = np.flatnonzero(d != 0)
    t = t[t > 300]
    return t, d[t], hi[t], lo[t]


def simulate(P, t, d, hi, lo, stop, tgt, H):
    h, l, c, atr, sp = P["h"], P["l"], P["c"], P["atr"], P["sp"]
    n = len(c)
    A = atr[t]
    entry = c[t]
    near, far = np.where(d == 1, lo, hi), np.where(d == 1, hi, lo)    # 自己這側 / 對邊
    buf = {"close": 0.25, "touch0": 0.0, "touch25": 0.25, "touch50": 0.5}[stop]
    sl = near - d * buf * A
    risk = (entry - sl) * d
    tg = {"中點": (hi + lo) / 2, "對邊": far, "1R": entry + d * risk, "2R": entry + d * 2 * risk}[tgt]
    ix = np.minimum(t[:, None] + np.arange(1, H + 1)[None, :], n - 1)
    fav = np.where(d[:, None] == 1, h[ix], -l[ix])
    cc = c[ix]
    if stop == "close":
        hsl = (cc * d[:, None]) < (near * d)[:, None]
    else:
        hsl = np.where(d[:, None] == 1, l[ix], -h[ix]) <= (sl * d)[:, None]
    htp = fav >= (tg * d)[:, None]
    big = 10 ** 6
    fsl = np.where(hsl.any(1), hsl.argmax(1), big)
    ftp = np.where(htp.any(1), htp.argmax(1), big)
    slpx = cc[np.arange(len(t)), np.minimum(fsl, H - 1)] if stop == "close" else sl
    px = cc[:, -1]
    px = np.where(ftp < fsl, tg, px)
    px = np.where((fsl <= ftp) & (fsl < big), slpx, px)
    bars = np.minimum(np.minimum(fsl, ftp), H - 1) + 1
    valid = ((tg - entry) * d > 0) & (risk > 0)
    R = ((px - entry) * d - sp[t]) / risk
    return R, bars, valid


def one_at_a_time(t, bars):
    keep, busy = np.zeros(len(t), bool), -1
    for i in range(len(t)):
        if t[i] > busy:
            keep[i] = True
            busy = t[i] + bars[i]
    return keep


def main():
    lines, rows, keep_trades = [], [], {}

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    data = {"M15": m15, "H1": resample(m15, "1h"), "M5": resample(load("data/XAUUSD_M1_2023_2026.csv"), "5min")}
    for tf, cfg in TFS.items():
        P = prep(data[tf], cfg["atr_n"])
        for L, zone, filt in itertools.product(LS, ZONES, FILTS):
            t, d, hi, lo = signals(P, L, zone, filt)
            for stop, tgt in itertools.product(STOPS, TGTS):
                R, bars, valid = simulate(P, t, d, hi, lo, stop, tgt, cfg["H"])
                tt, RR, bb = t[valid], R[valid], bars[valid]
                k = one_at_a_time(tt, bb)
                tt, RR = tt[k], RR[k]
                time = P["idx"][tt]
                res, val = RR[time < CUT], RR[time >= CUT]
                yr = pd.Series(RR).groupby(time.year).mean()
                sr, sv = summ(res), summ(val)
                ndays = pd.Index(time.normalize()).nunique()
                rows.append(dict(tf=tf, L=L, zone=zone, filt=filt, stop=stop, tgt=tgt, n=len(RR),
                                 per_day=len(RR) / max(ndays, 1), res=sr["mean"], res_t=sr["t"], val=sv["mean"],
                                 val_t=sv["t"], val_pf=sv["pf"], **{f"y{y}": v for y, v in yr.items()}))
                keep_trades[(tf, L, zone, filt, stop, tgt)] = (time, RR)
    G = pd.DataFrame(rows)
    G.to_csv("part24_grid.csv", index=False)

    out("# 第二十四部分：箱體邊緣反向（研究期 2023~2024 / 驗證期 2025~2026/10）")
    out(f"參數組：{len(G)}（週期 3 × 箱長 {LS} × 區域 {ZONES} × 過濾 {FILTS} × 停損 {STOPS} × 停利 {TGTS}），一次一筆、扣點差")
    for tf in TFS:
        g = G[G.tf == tf]
        out(f"\n## {tf}")
        out(f"  全部 {len(g)} 組：研究期平均 {g.res.mean():+.3f}R（為正 {(g.res > 0).mean():.0%}），"
            f"驗證期平均 {g.val.mean():+.3f}R（為正 {(g.val > 0).mean():.0%}）")
        for col in ("stop", "tgt", "L", "zone", "filt"):
            s = g.groupby(col)[["res", "val"]].mean()
            out(f"  按 {col:5s}: " + " | ".join(f"{k} {r.res:+.3f}/{r.val:+.3f}" for k, r in s.iterrows()))
        top = g.sort_values("res", ascending=False).head(10)
        out("  研究期前 10 名 → 驗證期：")
        for _, r in top.iterrows():
            yrs = " ".join(f"{y}:{r[f'y{y}']:+.2f}" for y in (2023, 2024, 2025, 2026) if f"y{y}" in r and pd.notna(r[f"y{y}"]))
            out(f"    L={r.L:2d} zone={r.zone} {r.filt:3s} 停損={r.stop:7s} 停利={r.tgt:3s} 每日 {r.per_day:4.1f} 筆 | "
                f"研究 {r.res:+.3f}R (t {r.res_t:+.2f}) → 驗證 {r.val:+.3f}R (t {r.val_t:+.2f}, PF {r.val_pf:.2f}) | {yrs}")
        out(f"  前 10 名驗證期為正：{(top.val > 0).sum()}/10，平均 {top.val.mean():+.3f}R")
        # 安慰劑原始定義（收盤停損 + 對邊停利）的毛 / 淨對照
        b = g[(g.stop == "close") & (g.tgt == "對邊") & (g.filt == "無")]
        out("  安慰劑原始定義（收盤破箱才出場、目標對邊、不過濾）：")
        for _, r in b.iterrows():
            out(f"    L={r.L:2d} zone={r.zone}: n={r.n:6d} 研究 {r.res:+.3f}R → 驗證 {r.val:+.3f}R")

    # 毛利（不扣點差）對照：用研究期最好的一組，看點差吃掉多少
    out("\n## 點差的影響（各週期研究期第一名）")
    for tf in TFS:
        g = G[G.tf == tf].sort_values("res", ascending=False).iloc[0]
        P = prep(data[tf], TFS[tf]["atr_n"])
        t, d, hi, lo = signals(P, g.L, g.zone, g.filt)
        P0 = dict(P, sp=np.zeros_like(P["sp"]))
        R0, bars, valid = simulate(P0, t, d, hi, lo, g.stop, g.tgt, TFS[tf]["H"])
        R1, _, _ = simulate(P, t, d, hi, lo, g.stop, g.tgt, TFS[tf]["H"])
        k = one_at_a_time(t[valid], bars[valid])
        out(f"  {tf}: 未扣點差 {R0[valid][k].mean():+.3f}R，扣點差 {R1[valid][k].mean():+.3f}R")
    with open("results_part24.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
