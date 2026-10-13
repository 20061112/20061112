"""第二十八部分：M1 急跌買的回測過擬合機率（PBO，Bailey et al. CSCV）。

把第二十二~二十六部分實際搜尋過的範圍重建成設定矩陣，取每組的「日損益」（R，固定每筆風險）：
  A 族（急跌買參數族）：門檻 {2,2.25,2.5,2.75,3} × 停損 {2,3,4,6} ATR × 持有 {24,48,120,240}
                        × span {10,20,30,40} × z 長度 {720,1380,2880} = 960 組
  B 族（A + 其他方向）：再加急漲追多 / 急漲做空 / 急跌追空 × 門檻 {2,2.5,3} × 停損 {2,3,6} × 持有 {24,120,240}
                        × span {10,20,40}（z 1380）= 243 組，共 1,203 組
CSCV：交易日依時間切成 S = 16 段，所有 8/8 組合（12,870 種）：樣本內挑日 Sharpe 最高 → 看樣本外百分位。
PBO = 樣本外百分位 ≤ 50% 的比例。另報告樣本內最佳的樣本外 Sharpe、定案設定的樣本外排名、績效衰減斜率。
輸出：results_part28.txt、part28_pbo.png
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

S = 16
FINAL = ("急跌買", 2.5, 3.0, 120, 20, 1380)
MAXH = 240
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=6)
dlt = c.diff().where(~gap)
o, h, l, cl, A = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), c.to_numpy(), atr.to_numpy()
spr = m.spread.to_numpy() * 0.01
ts = m.index.to_numpy()
tday = ((m.index - pd.Timedelta(hours=1)).normalize()).to_numpy()
START = 7500                                                     # 最長 z 長度 + ATR 暖機
SIGS = {"急跌買": (-1, 1), "急漲追多": (1, 1), "急漲做空": (1, -1), "急跌追空": (-1, -1)}   # (觸發方向, 交易方向)


def zcalc(span, zw):
    v = dlt.ewm(span=span, adjust=False).mean() / atr
    return ((v - v.rolling(zw).mean()) / v.rolling(zw).std()).to_numpy()


def cands(Z, zdir, thr):
    trig = (Z <= -thr) if zdir < 0 else (Z >= thr)
    pos = np.flatnonzero(trig & ~np.r_[False, trig[:-1]])
    pos = pos[(pos >= START) & (pos + MAXH + 1 < len(m))]
    return pos[np.isfinite(A[pos])]


def daily(pos, side, sl, hold):
    e = pos + 1
    idx = e[:, None] + np.arange(hold)[None, :]
    ep, sd = o[e], sl * A[pos]
    stop = ep - side * sd
    hit = (l[idx] <= stop[:, None]) if side == 1 else (h[idx] >= stop[:, None])
    any_ = hit.any(1)
    j = np.where(any_, hit.argmax(1), hold - 1)
    jj = idx[np.arange(len(e)), j]
    xp = np.where(any_, np.where(j > 0, (np.minimum if side == 1 else np.maximum)(o[jj], stop), stop), cl[jj])
    ok = (ts[pos + hold] - ts[pos]) <= np.timedelta64(int(hold * 1.6 + 30), "m")
    R = (side * (xp - ep) - spr[e]) / sd
    keep, busy = [], -1
    for k in range(len(pos)):                                    # 同時只持有一筆
        if ok[k] and pos[k] > busy:
            keep.append(k)
            busy = jj[k]
    keep = np.array(keep, int)
    return pd.Series(R[keep]).groupby(tday[pos[keep]]).sum()


cols = {}
zc = {}
grid_A = list(itertools.product(["急跌買"], (2.0, 2.25, 2.5, 2.75, 3.0), (2.0, 3.0, 4.0, 6.0), (24, 48, 120, 240), (10, 20, 30, 40), (720, 1380, 2880)))
grid_B = list(itertools.product(["急漲追多", "急漲做空", "急跌追空"], (2.0, 2.5, 3.0), (2.0, 3.0, 6.0), (24, 120, 240), (10, 20, 40), (1380,)))
for cfg in grid_A + grid_B:
    sig, thr, sl, hold, span, zw = cfg
    if (span, zw) not in zc:
        zc[(span, zw)] = zcalc(span, zw)
    zdir, side = SIGS[sig]
    cols[cfg] = daily(cands(zc[(span, zw)], zdir, thr), side, sl, hold)
days = np.unique(tday[START:])
M_all = pd.DataFrame(cols).reindex(days).fillna(0.0)
M_all = M_all.loc[:, :]                                          # 日 × 設定


def cscv(M, label):
    X = M.to_numpy()
    n, k = X.shape
    blocks = np.array_split(np.arange(n), S)
    bs = np.array([X[b].sum(0) for b in blocks])                 # 每段總和
    bq = np.array([(X[b] ** 2).sum(0) for b in blocks])
    bn = np.array([len(b) for b in blocks])
    fi = list(M.columns).index(FINAL) if FINAL in M.columns else None
    logits, is_best, oos_best, oos_med, rank_final, deg = [], [], [], [], [], []
    for ins in itertools.combinations(range(S), S // 2):
        mi = np.zeros(S, bool)
        mi[list(ins)] = True

        def sharpe(mask):
            N = bn[mask].sum()
            mu = bs[mask].sum(0) / N
            var = bq[mask].sum(0) / N - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-12)) * np.sqrt(252)
        si, so = sharpe(mi), sharpe(~mi)
        b = int(np.argmax(si))
        r = (so < so[b]).mean() + 0.5 * (so == so[b]).mean()
        r = min(max(r, 1e-6), 1 - 1e-6)
        logits.append(np.log(r / (1 - r)))
        is_best.append(si[b])
        oos_best.append(so[b])
        oos_med.append(np.median(so))
        if fi is not None:
            rank_final.append((so < so[fi]).mean())
        deg.append((si, so))
    logits, is_best, oos_best = map(np.array, (logits, is_best, oos_best))
    full = M.mean() / M.std() * np.sqrt(252)
    p(f"\n[{label}] {k} 組設定 × {n} 個交易日；S = {S} 段，{len(logits)} 種切法")
    p(f"  PBO（樣本內最佳在樣本外落到後半）= {(logits <= 0).mean():.1%}")
    p(f"  樣本內最佳：樣本內 Sharpe 平均 {is_best.mean():.2f} → 樣本外 Sharpe 平均 {oos_best.mean():.2f}（中位 {np.median(oos_best):.2f}），"
      f"樣本外為負的比例 {(oos_best < 0).mean():.1%}；同切法全部設定樣本外中位 {np.mean(oos_med):.2f}")
    p(f"  全期日 Sharpe：中位 {full.median():.2f}，最好 {full.max():.2f}，為負的組數 {(full < 0).sum()} / {k}")
    if fi is not None:
        rf = np.array(rank_final)
        p(f"  定案設定（門檻 2.5、停損 3、持有 120、span 20、z 1380）：全期 Sharpe {full[FINAL]:.2f}（第 {int((full > full[FINAL]).sum()) + 1} 名）；"
          f"樣本外百分位平均 {rf.mean():.0%}，落在前半的比例 {(rf > .5).mean():.0%}")
    # 衰減：樣本內 Sharpe 對樣本外 Sharpe 的迴歸斜率（全部設定、全部切法平均）
    slopes = [np.polyfit(si, so, 1)[0] for si, so in deg[::50]]
    p(f"  樣本內 → 樣本外 Sharpe 迴歸斜率（平均）{np.mean(slopes):+.2f}（1 = 完全延續，0 = 無關，< 0 = 反轉）")
    return logits, oos_best, full


p("=" * 110)
p("第二十八部分：M1 急跌買 PBO（CSCV）")
p("=" * 110)
MA = M_all[[c for c in M_all.columns if c[0] == "急跌買"]]
la, oa, fa = cscv(MA, "A 族：急跌買參數族")
lb, ob, fb = cscv(M_all, "B 族：A + 急漲追多 / 急漲做空 / 急跌追空")

p("\n--- A 族：全期日 Sharpe 依單一參數平均")
F = pd.DataFrame([dict(門檻=k[1], 停損=k[2], 持有=k[3], span=k[4], z長度=k[5], Sharpe=v) for k, v in fa.items()])
for col in ("門檻", "停損", "持有", "span", "z長度"):
    p(F.groupby(col).Sharpe.agg(["mean", "min", "max"]).round(2).T.to_string())
p(f"\n--- A 族 Sharpe 前 10 名")
p(F.nlargest(10, "Sharpe").round(2).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(la, bins=40, color="tab:blue", alpha=0.7)
ax[0].axvline(0, color="k", ls="--")
ax[0].set_title(f"A: logit of OOS rank (PBO {(la <= 0).mean():.1%})")
ax[1].hist(lb, bins=40, color="tab:orange", alpha=0.7)
ax[1].axvline(0, color="k", ls="--")
ax[1].set_title(f"B: logit of OOS rank (PBO {(lb <= 0).mean():.1%})")
ax[2].hist(fa.values, bins=40, color="tab:green", alpha=0.7)
ax[2].axvline(fa[FINAL], color="k", ls="--")
ax[2].set_title("A: full-period daily Sharpe of 960 configs (dashed = final)")
fig.tight_layout()
fig.savefig("part28_pbo.png", dpi=110)
open("results_part28.txt", "w").write("\n".join(out) + "\n")
