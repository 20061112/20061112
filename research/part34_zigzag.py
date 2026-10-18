"""第三十四部分：Zigzag 低點連成上升趨勢線，價格碰線時加倉。

只用「當時已確認」的轉折點（Zigzag 會重畫：低點要等收盤從低點反彈 ≥ th 才確認）。
趨勢線：最近 2 個已確認低點連線，或最近 3 個低點做對數價格回歸；只用斜率 > 0 的線（低點墊高）。
碰線：前一天最低價 ≤ 線 × (1 + tol) 且收盤 ≥ 線 × 0.97 → 今天收盤動作。
出場：收盤跌破線 3%，或持有滿 60 天。
A. 事件研究：碰線後 5 / 20 / 60 天報酬、跌破線的比例，對照「同樣在 MA200 之上、隨機日子」
B. 疊在策略上：策略原本 1 倍時拉到 2 或 3 倍（已 3 倍不變）；安慰劑 = 同次數隨機日子
C. 四個指數：12 組設定（th × 連線方式 × tol）的中位結果
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = open("results_part34.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def zigzag_lows(H, L, C, th):
    """回傳已確認低點：[(低點 index, 低點價格, 確認日 index)]。"""
    lows, mode = [], "down"
    ext, idx = L[0], 0
    for i in range(1, len(C)):
        if mode == "down":
            if L[i] < ext:
                ext, idx = L[i], i
            elif C[i] >= ext * (1 + th):
                lows.append((idx, ext, i)); mode = "up"; ext, idx = H[i], i
        else:
            if H[i] > ext:
                ext, idx = H[i], i
            elif C[i] <= ext * (1 - th):
                mode = "down"; ext, idx = L[i], i
    return lows


def trendline(n, lows, k):
    """line[i]：第 i 天收盤可用的趨勢線價格（只用確認日 ≤ i 的低點）；NaN = 沒有上升線。"""
    line = np.full(n, np.nan)
    j = 0
    conf = []
    for i in range(n):
        while j < len(lows) and lows[j][2] <= i:
            conf.append(lows[j]); j += 1
        if len(conf) < k:
            continue
        pts = conf[-k:]
        t = np.array([p[0] for p in pts], float); y = np.log([p[1] for p in pts])
        if k == 2:
            slope = (y[1] - y[0]) / (t[1] - t[0]); icpt = y[1] - slope * t[1]
        else:
            slope, icpt = np.polyfit(t, y, 1)
        if slope > 0:
            line[i] = np.exp(icpt + slope * i)
    return line


class Book:
    def __init__(self, name, fut, ratio, start=None):
        self.DF = DF = load(f"data/{name.lower()}_us_d.csv")
        self.S = S = signals(DF)
        self.i0 = 0 if start is None else S.index.searchsorted(pd.Timestamp(start))
        y0 = S.index[self.i0].year + 1
        self.win = list(range(y0, 2017))
        self.YS = year_starts(S, y0, 2016)
        self.kw = dict(eng="allfut", rebal="daily", fut_name=fut, ratio=ratio, L=3.0)
        self.split = pd.Timestamp("2013-01-01") if start is None else pd.Timestamp("2016-01-01")
        self.H, self.L, self.C = DF.high.values, DF.low.values, DF.close.values
        self.ON = S.on.values

    def signal(self, th, k, tol):
        lows = zigzag_lows(self.H, self.L, self.C, th)
        line = trendline(len(self.C), lows, k)
        touch = (self.L <= line * (1 + tol)) & (self.C >= line * 0.97)
        brk = self.C < line * 0.97
        return np.r_[False, touch[:-1]], np.r_[False, brk[:-1]], line, lows

    def overlay(self, trig, brk, extra, H=60):
        n = len(self.C)
        reg, Lr = self.ON.copy(), np.where(self.ON, 3.0, 1.0)
        left, cnt = 0, 0
        for i in range(n):
            if left == 0 and trig[i] and not self.ON[i]:
                left, cnt = H, cnt + 1
            if left > 0:
                if brk[i]:
                    left = 0
                    continue
                if not self.ON[i]:
                    reg[i], Lr[i] = True, 1.0 + extra
                left -= 1
        return reg, Lr, cnt

    def ev(self, windows=True, **extra):
        kw = {**self.kw, **extra}
        o = run(self.S, i0=self.i0, **kw)
        m = metrics(o)
        tw = pd.Series(o["twr"], index=self.S.index[self.i0:])
        m["h1"], m["h2"] = cagr(tw[tw.index < self.split]), cagr(tw[tw.index >= self.split])
        if windows:
            m["wt"] = np.array([metrics(run(self.S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["tot"] for y in self.win])
        return m

    def placebo(self, trig, brk, extra, act, n_draw=60):
        rng = np.random.default_rng(34)
        cand = np.flatnonzero(~self.ON & (np.arange(len(self.C)) > max(self.i0, 260)))
        k = max(int((trig & ~self.ON)[self.i0:].sum()), 1)
        vals = []
        for _ in range(n_draw):
            t = np.zeros(len(self.C), bool); t[rng.choice(cand, min(k, len(cand)), replace=False)] = True
            reg, Lr, _ = self.overlay(t, brk, extra)
            vals.append(self.ev(windows=False, regime_arr=reg, L_arr=Lr)["irr"])
        return (np.array(vals) < act).mean() * 100


CFG = list(itertools.product((0.05, 0.08, 0.12), (2, 3), (0.01, 0.02)))
P("第三十四部分：Zigzag 低點趨勢線，碰線加倉\n")

# ---------------------------------------------------------------- A. 事件研究（QQQ）
Q = Book("QQQ", "MNQ", 41.0)
C = pd.Series(Q.C, index=Q.S.index)
fwd = {h: C.shift(-h) / C - 1 for h in (5, 20, 60)}
above = (C > C.rolling(200).mean()).values
P("A. 事件研究（QQQ 1999 起；碰線日 = 前一天碰線、今天收盤進場，前瞻從今天收盤起算）")
P(f"  {'設定':22s}{'次數':>5s}{'5天':>8s}{'20天':>8s}{'60天':>8s}{'20天內跌破':>9s}│ 對照：MA200 之上隨機日 5 / 20 / 60 天")
base = {h: np.nanmean(fwd[h].values[above]) for h in fwd}
for th, k, tol in CFG:
    trig, brk, line, lows = Q.signal(th, k, tol)
    idx = np.flatnonzero(trig)
    # 同一段碰線只算第一次（10 天內重複的不重算）
    keep, last = [], -99
    for i in idx:
        if i - last > 10:
            keep.append(i); last = i
    keep = np.array(keep)
    if len(keep) == 0:
        continue
    f = {h: np.nanmean(fwd[h].values[keep]) for h in fwd}
    br = np.mean([brk[i:i + 20].any() for i in keep])
    P(f"  th {th:.0%} {k}點 tol {tol:.0%}     {len(keep):5d}{f[5] * 100:+7.2f}%{f[20] * 100:+7.2f}%{f[60] * 100:+7.2f}%{br * 100:8.0f}% │"
      f" {base[5] * 100:+.2f}% / {base[20] * 100:+.2f}% / {base[60] * 100:+.2f}%")

# ---------------------------------------------------------------- B. 疊在策略上（QQQ）
P("\nB. QQQ 策略（全期貨、每天重設）：策略在 1 倍時碰線 → 拉到 2 / 3 倍，跌破 3% 或 60 天出場")
B0 = Q.ev()
P(f"  基準 IRR {B0['irr'] * 100:.1f}%、最慘 {B0['worst'] * 100:.0f}%、回撤 {B0['mdd'] * 100:.0f}%")
P(f"  {'設定':26s}{'次數':>5s}{'IRR':>7s}{'最慘':>6s}{'回撤':>6s}{'前半差':>8s}{'後半差':>7s}{'勝段':>6s}{'安慰劑':>7s}")
rows = []
for (th, k, tol), extra in itertools.product(CFG, (1.0, 2.0)):
    trig, brk, *_ = Q.signal(th, k, tol)
    reg, Lr, cnt = Q.overlay(trig, brk, extra)
    m = Q.ev(regime_arr=reg, L_arr=Lr)
    pl = Q.placebo(trig, brk, extra, m["irr"]) if extra == 1.0 else np.nan
    rows.append(dict(th=th, k=k, tol=tol, extra=extra, d=m["irr"] - B0["irr"], pl=pl))
    P(f"  th {th:.0%} {k}點 tol {tol:.0%} → {1 + extra:.0f}倍{cnt:5d}{m['irr'] * 100:6.1f}%{m['worst'] * 100:6.0f}%{m['mdd'] * 100:6.0f}%"
      f"{(m['h1'] - B0['h1']) * 100:+7.1f}%{(m['h2'] - B0['h2']) * 100:+7.1f}%{(m['wt'] > B0['wt']).sum():3d}/17"
      + (f"{pl:6.0f}%" if pl == pl else ""))
R = pd.DataFrame(rows)
P(f"  2 倍 12 組：IRR 差中位 {R[R.extra == 1].d.median() * 100:+.1f}%、安慰劑中位 {R[R.extra == 1].pl.median():.0f}%；"
  f"3 倍 12 組：IRR 差中位 {R[R.extra == 2].d.median() * 100:+.1f}%")

# ---------------------------------------------------------------- C. 四個指數
P("\nC. 四個指數（12 組設定的中位；SPY / DIA / SOXX 2006/3 起，SOXX 假設有期貨）")
for name, fut, ratio, start in (("QQQ", "MNQ", 41.0, None), ("SPY", "MES", 10.0, "2006-03-01"),
                                ("DIA", "MYM", 100.0, "2006-03-01"), ("SOXX", "MNQ", 41.0, "2006-03-01")):
    Bk = Q if name == "QQQ" else Book(name, fut, ratio, start)
    b = Bk.ev(windows=False)
    d, w, h1, h2 = [], [], [], []
    for th, k, tol in CFG:
        trig, brk, *_ = Bk.signal(th, k, tol)
        reg, Lr, _ = Bk.overlay(trig, brk, 1.0)
        m = Bk.ev(windows=False, regime_arr=reg, L_arr=Lr)
        d.append(m["irr"] - b["irr"]); w.append(m["worst"] - b["worst"])
        h1.append(m["h1"] - b["h1"]); h2.append(m["h2"] - b["h2"])
    d, w, h1, h2 = map(np.array, (d, w, h1, h2))
    P(f"  {name:5s} 2 倍：IRR 差中位 {np.median(d) * 100:+.1f}%（{d.min() * 100:+.1f} ~ {d.max() * 100:+.1f}）｜IRR 變好 {(d > 0).sum()}/12｜"
      f"兩半都變好 {((h1 > 0.002) & (h2 > 0.002)).sum()}/12｜最慘變化中位 {np.median(w) * 100:+.0f} 點")

# ---------------------------------------------------------------- 圖
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
trig, brk, line, lows = Q.signal(0.08, 2, 0.02)
seg = slice(Q.S.index.searchsorted(pd.Timestamp("2023-01-01")), len(Q.C))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(C.index[seg], Q.C[seg], color="#444", lw=1, label="QQQ close")
ax.plot(C.index[seg], line[seg], color="#1f77b4", lw=1.2, label="Trendline (last 2 confirmed lows, as known that day)")
lw = [(C.index[i], p, C.index[c]) for i, p, c in lows if C.index[i] >= C.index[seg][0]]
ax.scatter([a for a, _, _ in lw], [p for _, p, _ in lw], color="#2ca02c", zorder=3, label="Zigzag low (8%)")
ax.scatter([c for _, _, c in lw], [Q.C[C.index.get_loc(c)] for _, _, c in lw], marker="x", color="#2ca02c", zorder=3,
           label="Date the low was confirmed")
tt = np.flatnonzero(trig[seg]) + seg.start
ax.scatter(C.index[tt], Q.C[tt], marker="^", color="#d62728", zorder=4, s=25, label="Touch -> add next close")
ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title("QQQ 2023-2026: zigzag 8% lows, rising trendline, touches")
plt.tight_layout(); plt.savefig("part34_zigzag.png", dpi=110)
out.close()
