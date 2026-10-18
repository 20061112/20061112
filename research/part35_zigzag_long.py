"""第三十五部分：長週期 Zigzag 趨勢線（週線，門檻 15~30%），碰線加倉。

週線（週五收）：低點要週收盤從低點反彈 ≥ th 才確認；最近 2 點連線 / 3 點回歸，只取上升線。
碰線：該週最低 ≤ 線 × (1 + tol) 且週收盤 ≥ 線 × 0.95 → 下週第一個交易日收盤動作。
出場：週收盤跌破線 5%，或持有滿 H 週。
A. 事件研究：碰線後 13 / 26 / 52 週報酬、跌破比例；對照 40 週均線之上的隨機週
B. 加槓桿：策略在 1 倍時拉到 2 倍，抱 26 / 52 週；安慰劑 = 同次數隨機日子
C. 加定投：碰線後 H 週內每週投入加倍（純定投與策略都測）
四個指數；設定 = th 15/20/25/30% × 2 / 3 點 × tol 2 / 5%（16 組）。
"""
import itertools
import numpy as np
import pandas as pd
from part22_core import load, signals, run, metrics, year_starts

out = None


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


def cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def zigzag_lows(H, L, C, th):
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
    line, conf, j = np.full(n, np.nan), [], 0
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
        self.ON = S.on.values
        W = DF.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        self.W = W
        # 每週訊號在下週第一個交易日生效
        self.wk_to_day = np.searchsorted(S.index.values, (W.index + pd.Timedelta(days=1)).values)

    def signal(self, th, k, tol):
        W = self.W
        H, L, C = W.high.values, W.low.values, W.close.values
        lows = zigzag_lows(H, L, C, th)
        line = trendline(len(C), lows, k)
        touch = (L <= line * (1 + tol)) & (C >= line * 0.95)
        brk = C < line * 0.95
        n = len(self.S)
        td, bd = np.zeros(n, bool), np.zeros(n, bool)
        for w in range(len(C)):
            d = self.wk_to_day[w]
            if d < n:
                td[d] |= bool(touch[w]); bd[d] |= bool(brk[w])
        return td, bd, touch, brk, line, lows

    def hold_mask(self, trig, brk, H_days):
        n, m, left = len(self.S), np.zeros(len(self.S), bool), 0
        cnt = 0
        for i in range(n):
            if left == 0 and trig[i]:
                left, cnt = H_days, cnt + 1
            if left > 0:
                if brk[i]:
                    left = 0
                    continue
                m[i] = True; left -= 1
        return m, cnt

    def ev(self, windows=True, eng="allfut", **extra):
        kw = {**self.kw, **extra} if eng == "allfut" else dict(eng="dca", **extra)
        o = run(self.S, i0=self.i0, **kw)
        m = metrics(o)
        tw = pd.Series(o["twr"], index=self.S.index[self.i0:])
        m["h1"], m["h2"] = cagr(tw[tw.index < self.split]), cagr(tw[tw.index >= self.split])
        if windows:
            m["wt"] = np.array([metrics(run(self.S, i0=self.YS[y], i1=self.YS[y + 10], **kw))["tot"] for y in self.win])
        return m


CFG = list(itertools.product((0.15, 0.20, 0.25, 0.30), (2, 3), (0.02, 0.05)))
BOOKS = [("QQQ", "MNQ", 41.0, None), ("SPY", "MES", 10.0, "2006-03-01"), ("DIA", "MYM", 100.0, "2006-03-01"),
         ("SOXX", "MNQ", 41.0, "2006-03-01")]


def main():
    global out
    out = open("results_part35.txt", "w")
    P("第三十五部分：長週期 Zigzag 趨勢線（週線）碰線加倉\n")

    for name, fut, ratio, start in BOOKS:
        Bk = Book(name, fut, ratio, start)
        W = Bk.W
        wc = W.close
        above40 = (wc > wc.rolling(40).mean()).values
        fwd = {h: (wc.shift(-h) / wc - 1).values for h in (13, 26, 52)}
        base = {h: np.nanmean(fwd[h][above40]) for h in fwd}
        P(f"===== {name}（{Bk.S.index[Bk.i0].date()} 起） =====")
        P(f"A. 事件研究（週）：對照 40 週線之上隨機週 13 / 26 / 52 週 {base[13] * 100:+.1f}% / {base[26] * 100:+.1f}% / {base[52] * 100:+.1f}%")
        P(f"  {'設定':20s}{'低點數':>5s}{'碰線次數':>6s}{'13週':>8s}{'26週':>8s}{'52週':>8s}{'26週內跌破':>9s}")
        for th, k, tol in CFG:
            _, _, touch, brk, line, lows = Bk.signal(th, k, tol)
            idx, keep, last = np.flatnonzero(touch), [], -99
            for i in idx:
                if i - last > 4:
                    keep.append(i); last = i
            keep = np.array(keep, int)
            if len(keep) == 0:
                P(f"  th {th:.0%} {k}點 tol {tol:.0%}   {len(lows):5d}{0:6d}")
                continue
            f = {h: np.nanmean(fwd[h][keep]) for h in fwd}
            br = np.mean([brk[i:i + 26].any() for i in keep])
            P(f"  th {th:.0%} {k}點 tol {tol:.0%}   {len(lows):5d}{len(keep):6d}{f[13] * 100:+7.1f}%{f[26] * 100:+7.1f}%{f[52] * 100:+7.1f}%{br * 100:8.0f}%")

        b_s = Bk.ev()
        b_d = Bk.ev(eng="dca")
        P(f"\nB. 加槓桿（策略 1 倍時 → 2 倍）｜基準 IRR {b_s['irr'] * 100:.1f}%、最慘 {b_s['worst'] * 100:.0f}%")
        P(f"C. 加定投（碰線後 H 週每週投入加倍）｜純定投基準 IRR {b_d['irr'] * 100:.1f}%")
        P(f"  {'設定':24s}{'次數':>4s}│{'B: IRR差':>8s}{'最慘差':>7s}{'前/後半':>13s}{'勝段':>6s}{'安慰劑':>6s}│{'C 策略: IRR差':>11s}{'C 定投: IRR差':>11s}")
        rows = []
        rng = np.random.default_rng(35)
        cand = np.flatnonzero(np.arange(len(Bk.S)) > max(Bk.i0, 260))
        for (th, k, tol), Hw in itertools.product(CFG, (26, 52)):
            trig, brk, *_ = Bk.signal(th, k, tol)
            mask, cnt = Bk.hold_mask(trig, brk, Hw * 5)
            if cnt == 0:
                continue
            lev_mask = mask & ~Bk.ON
            reg, Lr = Bk.ON | lev_mask, np.where(Bk.ON, 3.0, np.where(lev_mask, 2.0, 1.0))
            m = Bk.ev(regime_arr=reg, L_arr=Lr)
            vals = []
            for _ in range(30):
                t = np.zeros(len(Bk.S), bool); t[rng.choice(cand, cnt, replace=False)] = True
                mk, _ = Bk.hold_mask(t, brk, Hw * 5)
                lm = mk & ~Bk.ON
                vals.append(Bk.ev(windows=False, regime_arr=Bk.ON | lm, L_arr=np.where(Bk.ON, 3.0, np.where(lm, 2.0, 1.0)))["irr"])
            pl = (np.array(vals) < m["irr"]).mean() * 100
            mult = np.where(mask, 2.0, 1.0)
            cs = Bk.ev(windows=False, amt_mult=mult)
            cd = Bk.ev(windows=False, eng="dca", amt_mult=mult)
            rows.append(dict(d=m["irr"] - b_s["irr"], both=(m["h1"] - b_s["h1"] > 0.002) and (m["h2"] - b_s["h2"] > 0.002), pl=pl,
                             cs=cs["irr"] - b_s["irr"], cd=cd["irr"] - b_d["irr"], cnt=cnt))
            P(f"  th {th:.0%} {k}點 tol {tol:.0%} 抱{Hw}週{cnt:4d}│{(m['irr'] - b_s['irr']) * 100:+7.1f}%{(m['worst'] - b_s['worst']) * 100:+6.0f}點"
              f"{(m['h1'] - b_s['h1']) * 100:+6.1f}/{(m['h2'] - b_s['h2']) * 100:+5.1f}%{(m['wt'] > b_s['wt']).sum():3d}/{len(m['wt'])}{pl:5.0f}%│"
              f"{(cs['irr'] - b_s['irr']) * 100:+10.2f}%{(cd['irr'] - b_d['irr']) * 100:+11.2f}%")
        R = pd.DataFrame(rows)
        if len(R):
            P(f"  小結：加槓桿 IRR 差中位 {R.d.median() * 100:+.1f}%、兩半都變好 {R.both.sum()}/{len(R)}、安慰劑中位 {R.pl.median():.0f}%｜"
              f"加定投 策略 {R.cs.median() * 100:+.2f}%、純定投 {R.cd.median() * 100:+.2f}%｜碰線次數中位 {R.cnt.median():.0f}")
        P("")

    # 圖：QQQ 20% 週線
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Q = Book("QQQ", "MNQ", 41.0)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, (th, k) in zip(axes, ((0.20, 2), (0.15, 3))):
        _, _, touch, brk, line, lows = Q.signal(th, k, 0.05)
        W = Q.W
        ax.plot(W.index, W.close, color="#444", lw=1, label="QQQ weekly close")
        ax.plot(W.index, line, color="#1f77b4", lw=1.2, label=f"Trendline ({k} confirmed lows, as known that week)")
        ax.scatter([W.index[i] for i, _, _ in lows], [p for _, p, _ in lows], color="#2ca02c", zorder=3, label=f"Zigzag low ({th:.0%})")
        ti = np.flatnonzero(touch)
        ax.scatter(W.index[ti], W.close.values[ti], marker="^", color="#d62728", s=25, zorder=4, label="Touch")
        ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"QQQ weekly, zigzag {th:.0%}, {k}-point rising line")
    plt.tight_layout(); plt.savefig("part35_zigzag_weekly.png", dpi=110)
    out.close()


if __name__ == "__main__":
    main()
