"""第四十二部分：一個 IB 開幾單？可以重複進場嗎？突破失敗回到 IB 內可以反手嗎？

A. 現況統計：每個 IB 最多 1 單（3 小時窗內第一次收盤突破；若先往反方向突破，這個 IB 就不做）。每天筆數分佈。
B. 同一個 IB 停損 / 保本出場後，窗內再次收盤突破就再進（最多 2、3 次）。
C. 反手 1（假突破）：OW 多單進場後，收盤回到 IB 高之下（空單：回到 IB 低之上）→ 在該收盤反手做空；
   停損 = 進場後的最高點；出場：(a) 收盤 (b) 碰到 IB 另一側 (c) 2R 保本 + 收盤。原多單照規則繼續（只評估反手單本身）。
D. 反手 2（反向突破）：OW 的 IB 先 / 後往價值區方向突破（收盤 < IB 低 − 0.05ATR）→ 做空，停損 IB 高，2R 保本，收盤出場。
全部 2R 保本、收盤出場；每筆 損益÷ATR、勝率、PF；0.01 手（最多 5 張）美元。
"""
import numpy as np
import pandas as pd
from part27_ow import data, pf, PER, prev_profiles

out = open("results_part42.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def manage(t, O, H, L, C, i, s, e, st, bm, be_R=2.0, target=None):
    """回傳 (出場價, 出場索引, 出場類型)。"""
    R = s * (e - st); best = e; be = False
    for k in range(i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), k, ("保本" if be else "停損")
        if target is not None and ((H[k] >= target) if s == 1 else (L[k] <= target)):
            return (max(O[k], target) if s == 1 else min(O[k], target)), k, "目標"
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if be_R and not be and s * (best - e) >= be_R * R:
            be = True; st = e
    return C[-1], len(C) - 1, "收盤"


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def show(name, T, alldays, multi=False):
    if not len(T):
        P(f"  {name:36s} 0 筆"); return
    C5 = cap(T).sort_values("exit_time"); eq = C5.pnl.cumsum()
    dl = C5.groupby("day").pnl.sum().reindex(alldays, fill_value=0)
    cells = ("  " + "  ".join(f"{n} {x.u.mean() * 100:+.1f}%" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]] if len(x))) if multi else ""
    P(f"  {name:36s} {len(T):5d}筆 勝{(T.pnl > 0.05 * T.risk).mean() * 100:3.0f}% 每筆損益÷ATR {T.u.mean() * 100:+5.1f}% PF{pf(T.pnl):.2f} 總{T.pnl.sum():+6,.0f}美元 | "
      f"5張 {C5.pnl.sum():+6,.0f} 回撤{(eq - eq.cummax()).min():+6,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f}" + cells)


for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    PV = prev_profiles(days, ARR)
    base, rep2, rep3, fb = {"c": [], "ib": [], "be": []}, [], [], {"c": [], "ib": [], "be": []}
    base = []; rev = []; first_opp = 0; n_ib = 0
    for d in days:
        if d < pd.Timestamp("2023-03-01") or d not in PV or not np.isfinite(D.atr10.get(d, np.nan)):
            continue
        t, O, H, L, C, S, mins = ARR[d]; poc, vah, val = PV[d]; atr = D.atr10[d]
        for h in np.arange(2, 11, 0.5):
            m = (mins >= h * 60) & (mins < h * 60 + 60)
            if m.sum() < 60 / bm * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            s = 1 if ibl > vah else (-1 if ibh < val else 0)
            if s == 0:
                continue
            n_ib += 1
            w = np.where((mins >= h * 60 + 60) & (mins < h * 60 + 240))[0]
            up_l, dn_l = ibh + 0.05 * atr, ibl - 0.05 * atr
            go = w[(C[w] > up_l) if s == 1 else (C[w] < dn_l)]          # 順向突破
            bk = w[(C[w] < dn_l) if s == 1 else (C[w] > up_l)]          # 反向突破（往價值區）
            if len(bk) and (not len(go) or bk[0] < go[0]):
                first_opp += 1
            st0 = ibl if s == 1 else ibh
            mk = lambda i, side, e, st, kind, n, **kw: (lambda px, k, why: dict(
                day=d, h=h, kind=kind, n=n, side=side, t_in=t[i] , exit_time=t[k] + pd.Timedelta(minutes=bm), why=why,
                risk=side * (e - st), pnl=side * (px - e) - S[i - 1], u=(side * (px - e) - S[i - 1]) / atr))(*manage(t, O, H, L, C, i, side, e, st, bm, **kw))
            # 順向：現況 + 重複進場（停損 / 保本後窗內再突破）
            if len(go):
                n = 0; nxt = go[0]
                while nxt is not None and n < 3 and nxt + 1 < len(C):
                    n += 1
                    r = mk(nxt + 1, s, C[nxt], st0, "順", n)
                    base.append(r)
                    k_exit = np.searchsorted(t, r["exit_time"] - pd.Timedelta(minutes=bm))
                    if r["why"] == "收盤":
                        break
                    later = go[go > k_exit]
                    nxt = later[0] if len(later) else None
                # 反手 1：第一筆進場後，收盤回到 IB 內
                i0 = go[0] + 1
                back = np.where((C[i0:] < ibh) if s == 1 else (C[i0:] > ibl))[0]
                if len(back):
                    j = i0 + back[0]
                    if j + 1 < len(C) and mins[j] < 23 * 60:
                        ext = H[i0 - 1:j + 1].max() if s == 1 else L[i0 - 1:j + 1].min()
                        e = C[j]
                        if -s * (e - ext) > 0:
                            for kind, kw in [("反手1 收盤出場", dict(be_R=None)), ("反手1 目標 IB 另一側", dict(be_R=None, target=st0)),
                                             ("反手1 2R 保本", dict(be_R=2.0))]:
                                rev.append(mk(j + 1, -s, e, ext, kind, 1, **kw))
            # 反手 2：往價值區方向的突破（IB 另一側）
            if len(bk):
                j = bk[0]
                if j + 1 < len(C):
                    rev.append(mk(j + 1, -s, C[j], ibh if s == 1 else ibl, "反手2 反向突破（任何時候）", 1))
                    if not len(go) or j < go[0]:
                        rev.append(mk(j + 1, -s, C[j], ibh if s == 1 else ibl, "反手2 反向先突破", 1))
                    else:
                        rev.append(mk(j + 1, -s, C[j], ibh if s == 1 else ibl, "反手2 順向停損後", 1))
    B = pd.DataFrame(base); R_ = pd.DataFrame(rev)
    alldays = [d for d in days if d >= pd.Timestamp("2023-03-01")]
    B1 = B[B.n == 1].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
    P("\n" + "=" * 150 + f"\n[{tf}]")
    P(f"  A. OW 條件成立的 IB {n_ib} 個；3 小時內順向突破 → 開單 {(B.n == 1).sum()} 個（{(B.n == 1).sum() / n_ib * 100:.0f}%）；"
      f"先往反方向突破（這個 IB 就不做）{first_opp} 個（{first_opp / n_ib * 100:.0f}%）")
    pdv = B1.groupby("day").size()
    P(f"     每個 IB 最多 1 單；但每天 18 個 IB 重疊 → 有交易的日子每天 {pdv.mean():.1f} 筆（中位 {pdv.median():.0f}、最多 {pdv.max()}）；"
      f"同一根 K 棒多個 IB 同時突破只開 1 單：去重前 {(B.n == 1).sum()} → 去重後 {len(B1)}")
    P(f"     一天筆數分佈：" + "  ".join(f"{k}筆 {v}天" for k, v in pdv.value_counts().sort_index().items()))
    show("現況（每 IB 1 單、去重）", B1, alldays, tf == "M15")
    B2 = B[B.n <= 2].sort_values(["t_in", "h", "n"]).drop_duplicates(["t_in", "side"])
    B3 = B.sort_values(["t_in", "h", "n"]).drop_duplicates(["t_in", "side"])
    P(f"  B. 重複進場（第 2 次本身：{(B.n == 2).sum()} 筆 每筆 {B[B.n == 2].u.mean() * 100:+.1f}% ATR；第 3 次 {(B.n == 3).sum()} 筆 {B[B.n == 3].u.mean() * 100:+.1f}%）")
    show("每 IB 最多 2 次", B2, alldays, tf == "M15")
    show("每 IB 最多 3 次", B3, alldays, tf == "M15")
    P("  C/D. 反手單（只看反手單本身）")
    for kind in ["反手1 收盤出場", "反手1 目標 IB 另一側", "反手1 2R 保本", "反手2 反向突破（任何時候）", "反手2 反向先突破", "反手2 順向停損後"]:
        X = R_[R_.kind == kind].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]) if len(R_) else R_
        show(kind, X, alldays, tf == "M15")
    best = R_[R_.kind == "反手1 2R 保本"].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
    show("現況 + 反手1（2R 保本）合併", pd.concat([B1, best]).reset_index(drop=True), alldays, tf == "M15")
    opp = R_[R_.kind == "反手2 反向先突破"]
    both = pd.concat([B1, opp]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    show("現況 + 反手2（反向先突破）合併", both, alldays, tf == "M15")
    P(f"     兩邊每日損益相關：{B1.groupby('day').pnl.sum().reindex(alldays, fill_value=0).corr(opp.groupby('day').pnl.sum().reindex(alldays, fill_value=0)):+.2f}")
out.close()
