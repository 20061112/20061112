"""第二十四部分：滾動障礙百分位 + 停損（S / ATR）× RR 研究，順勢追突破。

障礙：ln(p_max/p_邊緣)（第二十三部分事件，range c=1 h=1 為主，tickvol c=1 h=0 為對照）。
滾動百分位：每個事件的 K 在「前 L 個交易日（不含當天）的所有事件」中的百分位；L = 20 / 60 / 120 / 擴張。
  → 不需要固定的前半門檻，全部年份都是前推的。
停損：S 倍數（0.5 / 0.75 / 1 / 1.5 / 2，S = √(D_eq·60)）或 M1 ATR60 倍數（3 / 5 / 8 / 12）。
停利：RR = 1 / 1.5 / 2 / 3 / 4，或不設停利（240 分鐘 / 交易日結束出場）。
模擬：M1 逐根；同一根同時碰到停損與停利算停損；扣點差；同一時間只持有一筆。
另做：(a) 停損 × RR 的巢狀前推（每年只用之前年份選 stop/RR）(b) 滑價壓力測試。
輸出 results_part24.txt、part24_grid.csv、part24_heatmap.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_kramers import load_all

H = 240
LS = [20, 60, 120, 0]                 # 0 = 擴張
QS = [0.8, 0.9, 0.95]
STOPS = [("S", k) for k in (0.5, 0.75, 1.0, 1.5, 2.0)] + [("ATR", k) for k in (3, 5, 8, 12)]
RRS = [1.0, 1.5, 2.0, 3.0, 4.0, None]
FEATS = ["range_c1.0_h1_sh", "tickvol_c1.0_h0_sh"]
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def cluster_t(R, day):
    R = np.asarray(R, float)
    r = pd.Series(R - R.mean()).groupby(np.asarray(day)).sum()
    se = np.sqrt((r ** 2).sum()) / len(R)
    return R.mean() / se if se > 0 else np.nan


d = load_all()
tr = np.maximum(d.high - d.low, np.maximum((d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()))
d["atr60"] = tr.rolling(60).mean()
DAYS = {k: g for k, g in d.groupby("day")}


def paths(E):
    """每個事件往後 H 根 M1 的有利 / 不利極值（相對進場價，往外為正）與收盤。"""
    n = len(E)
    fav = np.full((n, H), np.nan); adv = np.full((n, H), np.nan); cl = np.full((n, H), np.nan)
    atr = np.full(n, np.nan); j0s = np.zeros(n, int)
    for i, ev in enumerate(E.itertuples()):
        g = DAYS[ev.day]
        j0 = int(np.searchsorted(g.index.values, np.datetime64(ev.time_close)))
        j0s[i] = j0
        if j0 == 0 or j0 >= len(g):
            continue
        atr[i] = g.atr60.iloc[j0 - 1]
        seg = g.iloc[j0:j0 + H]
        s = ev.side
        hi, lo, c = seg.high.to_numpy() - ev.close, seg.low.to_numpy() - ev.close, seg.close.to_numpy() - ev.close
        m = len(seg)
        fav[i, :m] = hi if s == 1 else -lo
        adv[i, :m] = -lo if s == 1 else hi
        cl[i, :m] = s * c
    return fav, adv, cl, atr, j0s


def first_hit(mat, lvl):
    hit = mat >= lvl[:, None]
    idx = np.where(hit.any(1), hit.argmax(1), 10 ** 6)
    return idx


def outcomes(fav, adv, cl, risk, rr):
    """回傳每筆（未扣成本）的 R 與出場 bar 序號。"""
    n = len(risk)
    last = np.where(np.isnan(cl), -1, np.arange(H)).max(1)          # 最後一根有效
    i_stop = first_hit(np.nan_to_num(adv, nan=-1e9), risk)
    i_tgt = first_hit(np.nan_to_num(fav, nan=-1e9), rr * risk) if rr else np.full(n, 10 ** 6)
    R = np.empty(n); jx = np.empty(n, int)
    stop_first = (i_stop <= i_tgt) & (i_stop <= last)
    tgt_first = (i_tgt < i_stop) & (i_tgt <= last)
    tout = ~(stop_first | tgt_first)
    R[stop_first] = -1.0; jx[stop_first] = i_stop[stop_first]
    if rr:
        R[tgt_first] = rr; jx[tgt_first] = i_tgt[tgt_first]
    lc = cl[np.arange(n), np.maximum(last, 0)]
    R[tout] = lc[tout] / risk[tout]; jx[tout] = last[tout]
    return R, jx


def non_overlap(sel, day, j0, jx):
    keep = np.zeros(len(sel), bool)
    busy = {}
    for i in np.nonzero(sel)[0]:
        if busy.get(day[i], -1) >= j0[i]:
            continue
        keep[i] = True
        busy[day[i]] = j0[i] + jx[i]
    return keep


def rolling_pct(E, col, L):
    """K 在前 L 個交易日事件中的百分位（不含當天）。"""
    days = np.sort(E.day.unique())
    di = {dd: i for i, dd in enumerate(days)}
    E_di = E.day.map(di).to_numpy()
    x = E[col].to_numpy(float)
    out = np.full(len(E), np.nan)
    by_day = pd.Series(np.arange(len(E))).groupby(E_di).apply(list).to_dict()
    for k in range(len(days)):
        if k not in by_day:
            continue
        lo = 0 if L == 0 else k - L
        if lo < 0 or (L == 0 and k < 60):
            continue
        hist = np.concatenate([x[by_day[j]] for j in range(lo, k) if j in by_day])
        hist = np.sort(hist[np.isfinite(hist)])
        if len(hist) < 100:
            continue
        idx = by_day[k]
        out[idx] = np.searchsorted(hist, x[idx], side="right") / len(hist)
    return out


grid = []
for tf in (1, 3):
    E = pd.read_pickle(f"part23_events_M{tf}.pkl").reset_index(drop=True)
    E["year"] = E.time_close.dt.year
    fav, adv, cl, atr, j0 = paths(E)
    day = E.day.to_numpy()
    sp = E.spread.to_numpy()
    pct = {(f, L): rolling_pct(E, f, L) for f in FEATS for L in LS}
    common = np.isfinite(pct[(FEATS[0], 120)])           # 統一評估期間：L=120 有值以後
    P(f"\nM{tf}：{len(E)} 事件；評估期間 {E.time_close[common].min():%Y-%m-%d} 起（{common.sum()} 事件）")
    for stype, k in STOPS:
        risk = k * (E.S.to_numpy() if stype == "S" else atr)
        for rr in RRS:
            R0, jx = outcomes(fav, adv, cl, risk, rr)
            for f in FEATS:
                for L in LS:
                    for q in QS:
                        sel = common & (pct[(f, L)] >= q) & np.isfinite(risk) & (risk > 0)
                        keep = non_overlap(sel, day, j0, jx)
                        R = R0[keep] - sp[keep] / risk[keep]
                        yrs = E.year.to_numpy()[keep]
                        row = dict(tf=tf, feat=f.split("_")[0], L=L, q=q, stop=f"{stype}{k}", rr=rr if rr else 0,
                                   n=len(R), R=R.mean(), t=cluster_t(R, day[keep]),
                                   pf=R[R > 0].sum() / max(1e-9, -R[R < 0].sum()),
                                   risk_usd=np.median(risk[keep]),
                                   R_slip10=(R0[keep] - (sp[keep] + 0.10) / risk[keep]).mean(),
                                   R_slip30=(R0[keep] - (sp[keep] + 0.30) / risk[keep]).mean())
                        for y in (2023, 2024, 2025, 2026):
                            row[f"y{y}"] = R[yrs == y].mean() if (yrs == y).any() else np.nan
                        grid.append(row)
        # 基準：不篩選
        for rr in RRS:
            R0, jx = outcomes(fav, adv, cl, risk, rr)
            keep = non_overlap(common & np.isfinite(risk) & (risk > 0), day, j0, jx)
            R = R0[keep] - sp[keep] / risk[keep]
            yrs = E.year.to_numpy()[keep]
            row = dict(tf=tf, feat="none", L=-1, q=0, stop=f"{stype}{k}", rr=rr if rr else 0, n=len(R), R=R.mean(),
                       t=cluster_t(R, day[keep]), pf=R[R > 0].sum() / max(1e-9, -R[R < 0].sum()),
                       risk_usd=np.median(risk[keep]),
                       R_slip10=(R0[keep] - (sp[keep] + 0.10) / risk[keep]).mean(),
                       R_slip30=(R0[keep] - (sp[keep] + 0.30) / risk[keep]).mean())
            for y in (2023, 2024, 2025, 2026):
                row[f"y{y}"] = R[yrs == y].mean() if (yrs == y).any() else np.nan
            grid.append(row)

G = pd.DataFrame(grid)
G.to_csv("part24_grid.csv", index=False)
YC = ["y2023", "y2024", "y2025", "y2026"]
G["all4"] = (G[YC] > 0).all(axis=1)

P("\n" + "=" * 110)
P("1. 滾動百分位（L、q）— 對所有停損 × RR 取中位數（range c1 h1）")
for tf in (1, 3):
    g = G[(G.tf == tf) & (G.feat.isin(["range", "none"]))]
    P(f"\nM{tf}")
    P(g.groupby(["feat", "L", "q"])[["n", "R", "t"] + YC + ["all4"]].median().round(3).to_string())

P("\n" + "=" * 110)
P("2. 停損 × RR 熱圖（M1，range c1 h1，L=120，q=0.9）：每筆平均 R / t / 逐年")
sub = G[(G.tf == 1) & (G.feat == "range") & (G.L == 120) & (G.q == 0.9)]
P(sub.pivot_table(index="stop", columns="rr", values="R").round(3).to_string())
P("\nt 值")
P(sub.pivot_table(index="stop", columns="rr", values="t").round(2).to_string())
P("\n四年都 > 0")
P(sub.pivot_table(index="stop", columns="rr", values="all4").astype(int).to_string())
P("\n中位停損金額（USD）")
P(sub.groupby("stop").risk_usd.median().round(2).to_string())
P("\n對照：同停損 × RR、不篩選（全部邊緣事件都追）")
P(G[(G.tf == 1) & (G.feat == "none")].pivot_table(index="stop", columns="rr", values="R").round(3).to_string())
P("\n區域穩健：M1 range，跨 L × q（12 組）中位 R")
P(G[(G.tf == 1) & (G.feat == "range")].pivot_table(index="stop", columns="rr", values="R", aggfunc="median").round(3).to_string())
P("\nM1 tickvol c1 h0（對照算法），跨 L × q 中位 R")
P(G[(G.tf == 1) & (G.feat == "tickvol")].pivot_table(index="stop", columns="rr", values="R", aggfunc="median").round(3).to_string())
P("\nM3 range，跨 L × q 中位 R")
P(G[(G.tf == 3) & (G.feat == "range")].pivot_table(index="stop", columns="rr", values="R", aggfunc="median").round(3).to_string())

# 熱圖
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (tf, feat, ttl) in zip(axes, [(1, "range", "M1 barrier>=rolling p90 (L=120)"),
                                      (1, "none", "M1 no filter"), (3, "range", "M3 barrier>=rolling p90 (L=120)")]):
    s = G[(G.tf == tf) & (G.feat == feat) & ((G.L == 120) & (G.q == 0.9) | (G.feat == "none"))]
    pv = s.pivot_table(index="stop", columns="rr", values="R").reindex([f"{a}{b}" for a, b in STOPS])
    im = ax.imshow(pv.values, cmap="RdBu", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(["time" if c == 0 else c for c in pv.columns])
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("RR")
fig.colorbar(im, ax=axes, shrink=.8, label="mean R / trade")
plt.savefig("part24_heatmap.png", dpi=110, bbox_inches="tight")

P("\n" + "=" * 110)
P("3. 巢狀前推：每年只用之前年份（滾動百分位 L=120、q=0.9 固定）選停損 × RR（之前各年平均 R 最大），套用到該年")
for tf in (1, 3):
    s = G[(G.tf == tf) & (G.feat == "range") & (G.L == 120) & (G.q == 0.9)].copy()
    # 用年份平均近似（以筆數加權不必要，因為每年事件數相近）
    for y in (2025, 2026):
        prev = [c for c in YC if int(c[1:]) < y]
        s["score"] = s[prev].mean(axis=1)
        b = s.sort_values("score", ascending=False).iloc[0]
        P(f"  M{tf} {y}：用 {prev[0][1:]}~{prev[-1][1:]} 選出 stop={b.stop} rr={b.rr or '時間出場'}（之前平均 {b.score:+.3f}R）"
          f" → {y} 實際 {b[f'y{y}']:+.3f}R；同年所有組合中位 {s[f'y{y}'].median():+.3f}R")

P("\n" + "=" * 110)
P("4. 滑價壓力測試（每筆額外 $0.10 / $0.30），M1 range L=120 q=0.9")
P(sub[["stop", "rr", "n", "risk_usd", "R", "R_slip10", "R_slip30"]].sort_values(["stop", "rr"]).round(3).to_string(index=False))
open("results_part24.txt", "w").write("\n".join(lines) + "\n")
