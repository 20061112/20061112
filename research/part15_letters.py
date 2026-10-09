"""第十五部分 B：TPO 字母的應用 —— 進場當下「當天發展中剖面」的 Market Profile 指標，能否再篩出更好的單。

基準交易：②+（IB 60 分、每整點、觸價進場），用 part15_ib_params.signals。
剖面：當天 01:00 到進場前一根 M1（不含進場那根）的 TPO，字母長度 p = 5 / 15 / 30 分，tick 1 美元。
指標（都已按交易方向調整，+ = 對這筆單有利的那一邊）：
  rf      輪動因子 / 期數：每期 high 比前期高 +1、低 −1；low 同理（經典 Rotation Factor）
  otf     最近連續「單向延伸」的期數（做多：每期 low 都比前期高）
  imb     TPO 失衡：(POC 之上 − 之下) / 總 TPO
  poc_mig dPOC 相對當天開盤的位置 / ATR
  lvl_poc 進場價距 dPOC / ATR
  single  單印（只有 1 個 TPO 的列）占比
  tail    反方向極端的尾巴長度（做多看當天低點的單印列數）/ ATR，美元
  poor    突破那一側極端那一列的 TPO 數（≥2 = poor high/low，未完成拍賣）
  lvl_z   進場價那一列的 TPO 數在剖面內的 z（< 0 = 薄區 LVN）
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, D, SPLIT, exits
from part15_ib_params import signals


def dev_profile(H, L, mins, upto, p):
    per = mins[:upto] // p
    lo0 = np.floor(L[:upto].min()); hi0 = np.floor(H[:upto].max())
    n = int(hi0 - lo0) + 1
    diff = np.zeros(n + 1)
    ph, pl = [], []
    for k in np.unique(per):
        m = per == k
        a, b = int(np.floor(L[:upto][m].min()) - lo0), int(np.floor(H[:upto][m].max()) - lo0)
        diff[a] += 1; diff[b + 1] -= 1
        ph.append(H[:upto][m].max()); pl.append(L[:upto][m].min())
    return lo0, np.cumsum(diff)[:n], np.array(ph), np.array(pl)


def features(r, p):
    t, O, H, L, C, S = ARR[r.day]
    mins = (t - r.day).total_seconds().values // 60
    i = r.i
    lo0, cnt, ph, pl = dev_profile(H, L, mins, i, p)
    side, atr = r.side, r.atr
    prices = lo0 + np.arange(len(cnt)) + 0.5
    mx = cnt.max(); cand = np.where(cnt == mx)[0]
    poc_i = cand[np.argmin(np.abs(cand - (len(cnt) - 1) / 2))]; poc = prices[poc_i]
    tot = cnt.sum()
    imb = (cnt[poc_i + 1:].sum() - cnt[:poc_i].sum()) / tot
    rf = (np.sign(np.diff(ph)).sum() + np.sign(np.diff(pl)).sum()) / max(len(ph) - 1, 1)
    otf = 0
    for k in range(len(ph) - 1, 0, -1):
        ok = pl[k] > pl[k - 1] if side == 1 else ph[k] < ph[k - 1]
        if not ok:
            break
        otf += 1
    single = (cnt == 1).mean()
    # 反方向極端的尾巴：做多看最低處連續單印列
    arr = cnt if side == 1 else cnt[::-1]
    tail = 0
    for c in arr:
        if c == 1: tail += 1
        else: break
    poor = cnt[-1] if side == 1 else cnt[0]
    k = int(np.floor(r.lvl - lo0))
    k = min(max(k, 0), len(cnt) - 1)
    lvl_z = (cnt[k] - cnt.mean()) / (cnt.std() + 1e-9)
    return dict(rf=side * rf, otf=otf, imb=side * imb, poc_mig=side * (poc - O[0]) / atr, lvl_poc=side * (r.lvl - poc) / atr,
                single=single, tail=tail / atr, poor=poor, lvl_z=lvl_z)


if __name__ == "__main__":
    out = open("results_part15.txt", "a")
    def P(x=""): print(x); out.write(x + "\n"); out.flush()
    s, _ = signals(60, 60)
    X = exits(s).reset_index(drop=True)
    X["is"] = X.day < SPLIT
    P("\n" + "=" * 110)
    P(f"第十五部分 B：TPO 字母指標（基準 ②+，{len(X)} 筆）。每格：1~5 月每筆 | 6~10 月每筆（筆數），分界用 1~5 月五分位")
    allf = {}
    for p in (5, 15, 30):
        F = pd.DataFrame([features(r, p) for r in X.itertuples()])
        F.columns = [f"{c}_{p}" for c in F.columns]
        allf[p] = F
        P(f"\n--- 字母長度 {p} 分 ---")
        for f in F.columns:
            x = F[f]
            if x.nunique() <= 6:
                grp = x.clip(upper=x.quantile(.95)); cats = sorted(grp.dropna().unique())
                labs = [f"{c:g}" for c in cats]
            else:
                e = np.unique(np.nanquantile(x[X["is"]], [0, .2, .4, .6, .8, 1])); e[0], e[-1] = -np.inf, np.inf
                grp = pd.cut(x, e); cats = grp.cat.categories; labs = [f"{c.right:.2f}" for c in cats]
            cells = []
            for c, l in zip(cats, labs):
                a = X.pnl[(grp == c) & X["is"]]; b = X.pnl[(grp == c) & ~X["is"]]
                cells.append(f"≤{l}: {a.mean():+5.1f}|{b.mean():+5.1f}({len(a)}/{len(b)})")
            P(f"  {f:10s} " + "  ".join(cells))
    F = pd.concat([X] + list(allf.values()), axis=1)
    F.to_csv("part15_letter_trades.csv", index=False)
    out.close()
