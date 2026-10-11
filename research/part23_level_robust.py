"""第二十三部分：第二十二部分 (B)「回撤到小 K 中點掛單」的交易穩健度。

交易規則（多單；空單鏡像），全部只用當下已知資料：
  1. 訊號：前 10 根走了 >= K ATR、仍在高檔，本根全長 <= small ATR（同第二十二部分）
  2. 價位 L = 小 K 的中點（另測高點 / 低點）
  3. 等價格先離開 L 至少 away ATR（期間沒先跌破 L − stop），之後在 L 掛限價買單，最多等 W 根
  4. 停損 L − stop ATR，停利 L + tp ATR，最長持有 hold 分鐘，逾時以 M1 收盤平倉
  5. 成交那根 M1 只採計停損方向（保守）；同一分鐘同時碰停損停利算停損；每筆扣成交當下點差 + extra 美元
  R = stop ATR（以 R 計）

輸出：主規格的完整統計、逐年/逐季/多空、成本壓力、參數鄰近表、隨機價位對照、
     逐年前推（只用過去年份挑參數）、交易明細 part23_trades.csv。
"""
import itertools
import numpy as np
import pandas as pd
from part22_pause_bar import M1, T1, H1A, L1A, SP1, TF_MIN, resample, events

C1A = M1.close.values
RULES = {"M5": "5min", "M15": "15min", "H1": "60min"}
BARS = {tf: resample(r) for tf, r in RULES.items()}
EVT = {}


def get_events(tf, K, small):
    key = (tf, K, small)
    if key not in EVT:
        E = events(BARS[tf], K=K, small=small)
        E = E[E.small].copy()
        df = BARS[tf].loc[E.index]
        E["hi"], E["lo"] = df.high.values, df.low.values
        E["start"] = np.searchsorted(T1, (E.index + pd.Timedelta(minutes=TF_MIN[tf])).values)
        EVT[key] = E
    return EVT[key]


def sim(tf="M15", K=4.0, small=0.5, stop=0.5, tp=0.5, away=1.0, level="mid", shift=0.0,
        W=100, hold=1440, extra=0.0, shifts=None):
    E = get_events(tf, K, small)
    wmin = W * TF_MIN[tf]
    rows = []
    shifts = np.zeros(len(E)) + shift if shifts is None else shifts
    for (t, e), sh in zip(E.iterrows(), shifts):
        d, a, s = int(e.dir), e.atr, int(e.start)
        base = {"mid": e.mid, "hi": e.hi if d == 1 else e.lo, "lo": e.lo if d == 1 else e.hi}[level]
        L = base + d * sh * a
        # 換成「順勢方向為正」的座標：多單 x = price，空單 x = -price
        h, l = H1A[s:s + wmin], L1A[s:s + wmin]
        fav, adv = (h, l) if d == 1 else (-l, -h)
        x = d * L
        away_hit = np.flatnonzero(fav >= x + away * a)
        if not len(away_hit):
            continue
        k = away_hit[0]
        if (adv[:k + 1] <= x - stop * a).any():
            continue
        back = np.flatnonzero(adv[k + 1:] <= x)
        if not len(back):
            continue
        f = s + k + 1 + back[0]
        H, Lo = H1A[f:f + hold], L1A[f:f + hold]
        fav2, adv2 = (H.copy(), Lo) if d == 1 else (-Lo, -H)
        fav2[0] = -np.inf                                # 成交那根只算不利方向
        ih = np.flatnonzero(fav2 >= x + tp * a)
        il = np.flatnonzero(adv2 <= x - stop * a)
        ih = ih[0] if len(ih) else 10 ** 9
        il = il[0] if len(il) else 10 ** 9
        if il <= ih and il < 10 ** 9:
            r, n_ = -1.0, il
        elif ih < 10 ** 9:
            r, n_ = tp / stop, ih
        else:
            n_ = len(H) - 1
            r = d * (C1A[f + n_] - L) / (stop * a)
        cost = (SP1[f] + extra) / (stop * a)
        rows.append((pd.Timestamp(T1[f]), t, d, r - cost, cost, n_, a))
    return pd.DataFrame(rows, columns=["fill_time", "signal_time", "dir", "R", "cost_R", "minutes", "atr"])


def stats(T, label=""):
    r = T.R.values
    if len(r) < 2:
        return f"{label} n={len(r)}"
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    loss = (r < 0).astype(int)
    streak = max((len(list(g)) for k, g in itertools.groupby(loss) if k), default=0)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return (f"{label} n={len(r):4d} 勝率 {(r > 0).mean():.2f} 期望 {r.mean():+.3f}R t={r.mean() / r.std(ddof=1) * np.sqrt(len(r)):+.2f} "
            f"PF {gp / gl if gl else np.inf:.2f} 累計 {eq[-1]:+6.1f}R 最大回撤 {dd:5.1f}R 最長連虧 {streak}")


def non_overlap(T):
    """同一時間只持有一筆（依成交時間排序，持倉中出現的新單略過）。"""
    T = T.sort_values("fill_time")
    keep, busy = [], pd.Timestamp.min
    for i, row in T.iterrows():
        if row.fill_time >= busy:
            keep.append(i)
            busy = row.fill_time + pd.Timedelta(minutes=int(row.minutes) + 1)
    return T.loc[keep]


def bootstrap(r, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(r, size=(n, len(r))).mean(axis=1)
    return np.percentile(m, [2.5, 97.5]), (m <= 0).mean()


if __name__ == "__main__":
    PRIMARY = dict(K=4.0, small=0.5, stop=0.5, tp=0.5, away=1.0, level="mid")
    print("主規格（第二十二部分已看過的設定，非事前登記）：", PRIMARY)

    print("\n" + "=" * 110 + "\n## 1. 主規格完整統計")
    allT = []
    for tf in RULES:
        T = sim(tf, **PRIMARY); T["tf"] = tf
        allT.append(T)
        ci, p0 = bootstrap(T.R.values)
        print(stats(T, f"  {tf:3s}"), f"| 95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}] P(期望<=0)={p0:.2f}")
        print(stats(non_overlap(T), "      不重疊"))
        print(f"      多 {T[T.dir == 1].R.mean():+.3f}R ({(T.dir == 1).sum()})  空 {T[T.dir == -1].R.mean():+.3f}R ({(T.dir == -1).sum()})"
              f"  平均成本 {T.cost_R.mean():.3f}R  中位持有 {T.minutes.median():.0f} 分  中位 R ≈ {(T.atr * .5).median():.2f} 美元")
        yr = T.groupby(T.fill_time.dt.year).R
        print("      逐年：" + "  ".join(f"{y}: {g.mean():+.2f}R ({len(g)}, 累計 {g.sum():+.1f})" for y, g in yr))
        q = T.groupby(T.fill_time.dt.to_period("Q")).R.sum()
        print(f"      正報酬季度 {(q > 0).sum()}/{len(q)}：" + " ".join(f"{v:+.0f}" for v in q.values))
    A = pd.concat(allT).sort_values("fill_time")
    A.to_csv("part23_trades.csv", index=False)
    print(stats(A, "  三週期合併"))

    print("\n" + "=" * 110 + "\n## 2. 成本壓力（每筆額外加滑價，美元）")
    for tf in RULES:
        print(f"  {tf}: " + "  ".join(f"+{x}: {sim(tf, **PRIMARY, extra=x).R.mean():+.3f}R" for x in (0, 0.1, 0.2, 0.5)))

    print("\n" + "=" * 110 + "\n## 3. 隨機價位對照：同一批訊號，L 隨機平移 ±1 ATR 內（200 次）")
    rng = np.random.default_rng(1)
    for tf in RULES:
        real = sim(tf, **PRIMARY).R.mean()
        nE = len(get_events(tf, PRIMARY["K"], PRIMARY["small"]))
        null = np.array([sim(tf, **PRIMARY, shifts=rng.uniform(-1, 1, nE)).R.mean() for _ in range(200)])
        print(f"  {tf}: 小K中點 {real:+.3f}R  隨機價位平均 {null.mean():+.3f}R（5%~95%: {np.percentile(null, 5):+.3f} ~ {np.percentile(null, 95):+.3f}）"
              f"  隨機 >= 實際的比例 {(null >= real).mean():.2f}")

    print("\n" + "=" * 110 + "\n## 4. 參數鄰近（期望 R / 筆數）")
    for tf in RULES:
        print(f"\n  [{tf}] 列：K、小K門檻；欄：停損=停利 X ATR")
        Xs = (0.3, 0.5, 0.75, 1.0)
        print("     " + "".join(f"{'X=' + str(x):>16s}" for x in Xs))
        for K, sm in itertools.product((3.0, 4.0, 5.0), (0.35, 0.5, 0.7)):
            cells = []
            for x in Xs:
                T = sim(tf, K=K, small=sm, stop=x, tp=x)
                cells.append(f"{T.R.mean():+.3f} ({len(T):4d})" if len(T) else "   -")
            print(f"     K={K} 小K<={sm}: " + "".join(f"{c:>16s}" for c in cells))
        print(f"  [{tf}] 其他維度（主規格只改一項）")
        for name, kw in [("價位=小K高點(順勢端)", dict(level="hi")), ("價位=小K低點(逆勢端)", dict(level="lo")),
                         ("離開 0.5 ATR", dict(away=0.5)), ("離開 2 ATR", dict(away=2.0)),
                         ("停利 0.5R", dict(tp=0.25)), ("停利 2R", dict(tp=1.0)), ("停損1ATR停利0.5", dict(stop=1.0, tp=0.5))]:
            T = sim(tf, **{**PRIMARY, **kw})
            print(f"     {name:22s} {T.R.mean():+.3f}R  n={len(T):4d}  勝率 {(T.R > 0).mean():.2f}")

    print("\n" + "=" * 110 + "\n## 5. 逐年前推：只用之前年份挑「K × 小K門檻 × X」最佳組，套用到下一年")
    grid = list(itertools.product((3.0, 4.0, 5.0), (0.35, 0.5, 0.7), (0.3, 0.5, 0.75, 1.0)))
    for tf in RULES:
        res = {g: sim(tf, K=g[0], small=g[1], stop=g[2], tp=g[2]) for g in grid}
        line, oos = [], []
        for y in (2024, 2025, 2026):
            def score(g):
                T = res[g]; r = T[T.fill_time.dt.year < y].R
                return r.mean() / r.std() * np.sqrt(len(r)) if len(r) >= 20 else -np.inf
            best = max(grid, key=score)
            T = res[best]; r = T[T.fill_time.dt.year == y].R
            oos.append(r)
            line.append(f"{y}: 選 K{best[0]}/{best[1]}/X{best[2]} → {r.mean():+.3f}R ({len(r)})")
        o = pd.concat(oos)
        print(f"  {tf}: " + " | ".join(line) + f"  || 前推合計 {o.mean():+.3f}R t={o.mean() / o.std() * np.sqrt(len(o)):+.2f} n={len(o)}")
        allg = np.array([res[g].R.mean() for g in grid])
        print(f"       全部 {len(grid)} 組參數：期望 > 0 的比例 {(allg > 0).mean():.2f}，中位 {np.median(allg):+.3f}R")

    print("\n" + "=" * 110 + "\n## 6. 同背景下「任何一根 K 線的中點」做同樣交易（小K 是否有加分）")
    for tf in RULES:
        for K in (3.0, 4.0, 5.0):
            S = sim(tf, K=K, small=0.5, stop=0.5, tp=0.5)
            Aall = sim(tf, K=K, small=99, stop=0.5, tp=0.5)
            print(f"  {tf} K={K}: 小K中點 {S.R.mean():+.3f}R ({len(S)})   任何K中點 {Aall.R.mean():+.3f}R ({len(Aall)}, "
                  f"多 {Aall[Aall.dir == 1].R.mean():+.3f} / 空 {Aall[Aall.dir == -1].R.mean():+.3f}; "
                  + " ".join(f"{y}:{g.mean():+.2f}" for y, g in Aall.groupby(Aall.fill_time.dt.year).R) + ")")
