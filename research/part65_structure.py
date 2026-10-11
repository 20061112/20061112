"""第六十五部分：核心的結構延伸 —— 失敗突破反手、獲利拆解 → results_part65.txt

1. 失敗突破反手：核心單停損（價格回到 IB 另一側）後，若之後（同一天、IB 結束後 6 小時內）M5 收盤再突破 IB 另一側 + 0.05 ATR，
   就順這個新方向進場；停損 = 原先突破方向的極值（多單 = 進場後到反手前的最高點… 這裡簡化為原 IB 的另一側 + 原突破段的極值），
   實作：反手單停損放在「第一次突破後到反手訊號之間」的反向極值外 0.05 ATR；收盤出場。
2. 獲利拆解（核心 + peff 濾網 + 每天每方向 ≤ 3）：依 IB 起點小時、進場小時、星期、多空、IB 寬度 / ATR 分組，
   研究期三段與 2026 分開列，找出各段都穩定的部分。
"""
import numpy as np
import pandas as pd
from core_ib import events, simulate, report, ARR, PEFF_MAX, SUBS, RESEARCH, VALID

pd.set_option("display.width", 260)


def cap_stack(X, n):
    X = X.sort_values("t_in")
    return X[X.groupby(["day", "side"]).cumcount() < n]


def reversal_events(E, X, win_after=360, buf=0.05):
    """核心單停損後的反手事件。"""
    out = []
    stopped = X[X.why == "停損"]
    for r in stopped.itertuples():
        t, mins, O, H, L, C, S = ARR[r.day]
        s = -r.side                                   # 反手方向
        j = int(r.j)                                  # 原單停損那根
        lvl_break = r.ibh + buf * r.atr if s == 1 else r.ibl - buf * r.atr
        w = np.flatnonzero((np.arange(len(C)) > j) & (mins < r.s0 + 60 + win_after) & ((mins + 1) % 5 == 0))
        hit = w[(C[w] > lvl_break) if s == 1 else (C[w] < lvl_break)]
        if not len(hit):
            continue
        i = hit[0]
        seg = slice(r.i0, i + 1)
        ext = L[seg].min() if s == 1 else H[seg].max()       # 原突破段的反向極值
        out.append(dict(day=r.day, s0=r.s0, i=i, i0=min(i + 1, len(C) - 1), side=s, lvl=C[i],
                        ibh=ext if s == -1 else r.ibh, ibl=ext if s == 1 else r.ibl, atr=r.atr, t_in=t[i], spread=S[i],
                        min_in=mins[i], peff=r.peff, pdir=r.pdir, trend20=r.trend20, eff20=r.eff20))
    R = pd.DataFrame(out)
    R["aligned"] = R.side == R.pdir
    return R.sort_values("t_in").drop_duplicates(["t_in", "side"]).reset_index(drop=True)


def tab(X, key, labels=None):
    rows = []
    X = X[X.peff <= PEFF_MAX]
    for lab, a, b in SUBS + [("研究合計", *RESEARCH), ("2026", *VALID)]:
        Y = X[(X.day >= a) & (X.day < b)]
        for g, Z in Y.groupby(key, observed=True):
            p = Z.pnl
            rows.append(dict(期間=lab, 組=g, PF=round(p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9), 2), n=len(p)))
    T = pd.DataFrame(rows).pivot(index="組", columns="期間", values="PF")
    N = pd.DataFrame(rows).pivot(index="組", columns="期間", values="n")
    T = T[[c for c in ["24/11~12", "25H1", "25H2", "研究合計", "2026"] if c in T]]
    T["研究筆數"] = N["研究合計"]
    T["四段都>1"] = (T[["24/11~12", "25H1", "25H2", "2026"]] > 1).all(axis=1).map({True: "✓", False: ""})
    return T


def main():
    out = []
    P = out.append
    E = events()
    X = simulate(E)
    P("=== 1. 失敗突破反手 ===")
    RV = reversal_events(X[X.peff <= PEFF_MAX], X)
    Y = simulate(RV)
    P("核心（不限疊單）：" + str(report(X)))
    P("反手單本身：" + str(report(Y)))
    both = pd.concat([X, Y])
    P("核心 + 反手：" + str(report(both)))

    P("\n=== 2. 獲利拆解（核心 + 每天每方向 ≤ 3；PF）===")
    C3 = cap_stack(X, 3)
    C3["IB起點"] = (C3.s0 // 60).astype(int).map(lambda h: f"{h:02d}:xx")
    C3["進場時段"] = pd.cut(C3.min_in // 60, [0, 8, 12, 16, 24], right=False, labels=["01~07", "08~11", "12~15", "16~23"])
    C3["星期"] = C3.day.dt.dayofweek.map(dict(enumerate("一二三四五")))
    C3["方向"] = C3.side.map({1: "多", -1: "空"})
    C3["IB寬/ATR"] = pd.cut((C3.ibh - C3.ibl) / C3.atr, [0, 0.15, 0.25, 0.4, 9], labels=["<0.15", "0.15~0.25", "0.25~0.4", ">0.4"])
    C3["出場原因"] = C3.why
    for k in ("IB起點", "進場時段", "星期", "方向", "IB寬/ATR"):
        P(f"\n[{k}]")
        P(tab(C3, k).to_string())
    txt = "\n".join(out)
    open("results_part65.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
