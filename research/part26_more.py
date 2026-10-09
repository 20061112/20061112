"""第二十六部分：時間出場、每筆美元損益、大週期同向加倉、逆勢（盤整區）測試 → results_part26.txt

主體同第二十五部分（A/B/C 三組 ER 回檔順勢、不做星期一）。
1. 出場：tp ∈ {1.5R, 2R, 不設} × 最長持有 ∈ {L/4, L/2, L, 2L, 4L} 根（到時收盤平倉）
2. 每筆美元損益（每盎司 = 0.01 手），已扣點差
3. 大週期同向加倉（H1 / H4 已收完 K 棒的 ER20 與方向同向）
   size  : 同向時下 2 單位、其他 1 單位（同一個進場點）
   add1R : 先下 1 單位；價格到 +1R 且大週期同向 → 加 1 單位，加碼單停損 = 原進場價、目標 = 原目標
   比較用「每單位風險」報酬，以及 P/MDD、Sharpe（與部位大小無關的比率）
4. 逆勢：盤整（|ER長| ≤ r）時，價格 S 根內急衝（|ER短| ≥ q）且創 N 根新高/低 → 反向進場
   停損 = 那個新高/低外 0.1 ATR，tp 1R / 1.5R，最長 L 根；對照：隨機 K 棒、隨機方向、同樣停損規則
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, SPLIT
from part24_er_cascade import ser
from part25_exits import CFGS, prep, er, entries, metrics
from part25_ext import tf_er_done


def sim(B, k, s, M, tp, cap, buf=0.1, add=None):
    """回傳 dict(R, j, addR)。tp=None 表示不設停利。add=True 時在 +1R 加碼。"""
    o, h, l, c, A, spr = B["o"], B["h"], B["l"], B["c"], B["A"], B["spr"]
    n = len(c)
    e = k + 1
    ext = l[k - M + 1:k + 1].min() if s == 1 else h[k - M + 1:k + 1].max()
    stop = ext - s * buf * A[k]
    ep = o[e]
    R = s * (ep - stop)
    if not R > 0:
        return None
    tgt = ep + s * tp * R if tp else None
    end = min(e + cap, n)
    added, aep, astop = False, None, None
    addR = 0.0
    for j in range(e, end):
        if added and ((l[j] <= astop) if s == 1 else (h[j] >= astop)):   # 加碼單停損較近，先檢查
            ax = min(astop, o[j]) if s == 1 else max(astop, o[j])
            addR = (s * (ax - aep) - spr[e]) / R
            added = None                       # 加碼單已出場
        if (l[j] <= stop) if s == 1 else (h[j] >= stop):
            xp = stop if j == e else (min(stop, o[j]) if s == 1 else max(stop, o[j]))
            break
        if tgt is not None and ((h[j] >= tgt) if s == 1 else (l[j] <= tgt)):
            xp = tgt
            break
        if add and added is False and ((h[j] >= ep + R) if s == 1 else (l[j] <= ep - R)):
            added, aep, astop = True, ep + s * R, ep
    else:
        j = end - 1
        xp = c[j]
    if added:                                   # 加碼單跟主單一起出場
        addR = (s * (xp - aep) - spr[e]) / R
    return dict(R=(s * (xp - ep) - spr[e]) / R, j=j, usd=s * (xp - ep) - spr[e], Rusd=R, addR=addR,
                added=added is not False)


def main():
    m1 = load_m1()
    days = sorted(set(m1.index.date))
    out = []
    P = out.append
    cache = {}
    sigs = {}
    for name, (tf, L, M, S, a, cool) in CFGS.items():
        B = cache.setdefault(tf, prep(m1, tf))
        if "h1" not in B:
            B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
        t, side = entries(B, L, M, S, a, cool)
        keep = B["idx"][t].dayofweek != 0
        sigs[name] = (B, t[keep], side[keep], L, M)

    def book(name, tp, cap, add=False):
        B, t, side, L, M = sigs[name]
        rows = []
        for k, s in zip(t, side):
            r = sim(B, k, s, M, tp, cap, add=add)
            if r:
                rows.append(dict(leg=name, time=B["idx"][k], xtime=B["idx"][r["j"]], side=s, bars=r["j"] - k,
                                 h1=s * B["h1"][k], h4=s * B["h4"][k], **r))
        return pd.DataFrame(rows)

    P("=== 1. 時間出場：tp × 最長持有（不做星期一）===")
    rows = []
    for name, (tf, L, M, S, a, cool) in CFGS.items():
        for tp in (1.5, 2, None):
            for cap in (L // 4, L // 2, L, 2 * L, 4 * L):
                T = book(name, tp, cap)
                m = metrics(T, days)
                rows.append(dict(設定=name, tp=tp or "無", 最長=f"{cap} 根", n=m["n"], avgR=m["avgR"], win=m["win"],
                                 PF=m["PF"], P_MDD=m["P_MDD"], Sharpe=m["Sharpe"], 前半=m["前半"], 後半=m["後半"],
                                 持有=m["平均持有"]))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 2. 每筆美元損益（每盎司 = 0.01 手，已扣點差；tp2、最長 2L）===")
    base = pd.concat([book(n, 2, 2 * CFGS[n][1]) for n in CFGS])
    for name, g in base.groupby("leg"):
        P(f"{name}: {len(g)} 筆、平均 ${g.usd.mean():+.2f}、中位 ${g.usd.median():+.2f}、平均獲利 ${g.usd[g.usd > 0].mean():+.2f}、"
          f"平均虧損 ${g.usd[g.usd < 0].mean():+.2f}、1R 平均 ${g.Rusd.mean():.1f}、總計 ${g.usd.sum():+.0f}")
    P(f"合計：{len(base)} 筆、平均 ${base.usd.mean():+.2f}/筆、總計 ${base.usd.sum():+.0f}（每盎司）；"
      f"每天約 {len(base) / len(days):.2f} 筆、${base.usd.sum() / len(days):+.1f}/天")
    P("基準指標：" + str(metrics(base, days)))

    P("\n=== 3. 大週期同向時加倉 ===")
    base = base.assign(both=(base.h1 > 0) & (base.h4 > 0))
    P(f"H1 且 H4 同向：{base.both.sum()} 筆 avgR {base.R[base.both].mean():+.3f}；其他 {(~base.both).sum()} 筆 "
      f"avgR {base.R[~base.both].mean():+.3f}")

    def ratio(T, w):
        T = T.assign(R=T.R * w)
        m = metrics(T, days)
        return dict(總R=m["總R"], 平均風險單位=round(float(w.mean()), 2), 每單位風險R=round(m["總R"] / w.sum(), 3),
                    PF=m["PF"], MDD=m["MDD"], P_MDD=m["P_MDD"], Sharpe=m["Sharpe"], 前半=m["前半"], 後半=m["後半"])

    rows = [dict(方式="固定 1 單位", **ratio(base, np.ones(len(base))))]
    for cond, lab in ((base.h4 > 0, "H4 同向"), (base.both, "H1+H4 同向"), ((base.h1 > 0.2) & (base.h4 > 0.2), "H1+H4 > 0.2")):
        rows.append(dict(方式=f"{lab} → 2 單位", **ratio(base, np.where(cond, 2.0, 1.0))))
        rows.append(dict(方式=f"{lab} → 1.5 單位", **ratio(base, np.where(cond, 1.5, 1.0))))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n加碼（到 +1R 才加 1 單位，加碼單停損 = 原進場價，目標 = 原 2R 目標）")
    addT = pd.concat([book(n, 2, 2 * CFGS[n][1], add=True) for n in CFGS])
    addT = addT.assign(both=(addT.h1 > 0) & (addT.h4 > 0))
    rows = []
    for lab, cond in (("不加碼", np.zeros(len(addT), bool)), ("全部都加", np.ones(len(addT), bool)),
                      ("H4 同向才加", (addT.h4 > 0).to_numpy()), ("H1+H4 同向才加", addT.both.to_numpy())):
        T = addT.assign(R=addT.R + np.where(cond, addT.addR, 0.0))
        m = metrics(T, days)
        add_n = int((cond & addT.added).sum())
        rows.append(dict(方式=lab, 加碼次數=add_n,
                         加碼單平均R=round(float(addT.addR[cond & addT.added].mean()), 3) if add_n else np.nan,
                         總R=m["總R"], PF=m["PF"], MDD=m["MDD"], P_MDD=m["P_MDD"], Sharpe=m["Sharpe"],
                         前半=m["前半"], 後半=m["後半"]))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 4. 逆勢：盤整中的急衝 → 反向（tp 1R / 1.5R，最長 L 根）===")
    rng = np.random.default_rng(3)
    rows = []
    for tf, L, S, N in (("5min", 20, 3, 20), ("15min", 20, 3, 20), ("15min", 40, 5, 40), ("30min", 20, 3, 20)):
        B = cache.setdefault(tf, prep(m1, tf))
        eL, eS = er(B, L), er(B, S)
        h, l, c = B["h"], B["l"], B["c"]
        hiN = pd.Series(h).rolling(N).max().to_numpy()
        loN = pd.Series(l).rolling(N).min().to_numpy()
        for r_ in (0.1, 0.2, 1.0):
            for q in (0.6, 0.8):
                with np.errstate(invalid="ignore"):
                    up = (np.abs(eL) <= r_) & (eS >= q) & (h >= hiN)
                    dn = (np.abs(eL) <= r_) & (eS <= -q) & (l <= loN)
                t, last = [], -10 ** 9
                for k in np.flatnonzero(up | dn):
                    if k - last > S and k > 120 and k < len(c) - 2:
                        t.append(k)
                    last = k
                t = np.array(t, int)
                side = np.where(up[t], -1, 1)
                for tp in (1, 1.5):
                    res = [sim(B, k, s, S + 1, tp, L) for k, s in zip(t, side)]
                    T = pd.DataFrame([dict(time=B["idx"][k], xtime=B["idx"][x["j"]], R=x["R"], bars=x["j"] - k)
                                      for k, x in zip(t, res) if x])
                    m = metrics(T, days) if len(T) > 5 else {}
                    rows.append(dict(週期=tf, L=L, S=S, 盤整ERL上限=("不限" if r_ == 1.0 else r_), 急衝ERS下限=q, tp=tp,
                                     n=m.get("n"), avgR=m.get("avgR"), PF=m.get("PF"), 前半=m.get("前半"),
                                     後半=m.get("後半")))
        # 隨機基準
        rt = rng.integers(200, len(c) - L - 2, 5000)
        rs = rng.choice([-1, 1], 5000)
        for tp in (1, 1.5):
            R = [sim(B, k, s, S + 1, tp, L) for k, s in zip(rt, rs)]
            v = np.array([x["R"] for x in R if x])
            rows.append(dict(週期=tf, L=L, S=S, 盤整ERL上限="隨機", 急衝ERS下限="-", tp=tp, n=len(v),
                             avgR=round(float(v.mean()), 3), PF=round(float(v[v > 0].sum() / -v[v < 0].sum()), 2)))
    P(pd.DataFrame(rows).to_string(index=False))
    open("results_part26.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
