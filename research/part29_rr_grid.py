"""第二十九部分：M15 突破在過濾 A / A+B 下，不同停損（s × ATR15）與報酬風險比（t = 停利 / 停損）的影響。

A = 順 50 日均線；B = 今天開盤到進場價沒有逆方向走超過 0.17 日 ATR（第二十八部分）。
停損 s = 0.5 / 1 / 1.5 / 2 / 3 ATR15；RR t = 1 / 1.5 / 2 / 3 / 4 / 5 / 收盤（不設停利，當天收盤出）。
同根順序用 K 棒顏色推估；每筆扣點差；美元 = 每盎司。疊單：不限 / 同方向一次 1 筆。
訓練 2023/6~2024/12、驗證 2025/1~2026/10。
需要先跑 part28_m15_filters.py（產生 part28_features.pkl）。
"""
import numpy as np
import pandas as pd
import part27_wide_brackets as W
from part27_m15_trades import sim_detail


def cap(df, n):
    """同方向同時最多 n 筆（同 part28_combo.cap）。"""
    if n is None:
        return df
    keep, open_ = [], {1: [], -1: []}
    for r in df.itertuples():
        lst = [x for x in open_[r.dir] if x > r.進場時間]
        if len(lst) < n:
            lst.append(r.出場時間)
            keep.append(r.Index)
        open_[r.dir] = lst
    return df.loc[keep]

IDX = W.B.d.index
F = pd.read_pickle("part28_features.pkl")
F["key"] = list(zip(F.i, F.dir))
feat = F.set_index("key")[["trend50", "day_move"]]
ev = W.events(15)
keys = [(i, side) for i, L, side, a in ev]
fe = feat.reindex(keys)
mA = (fe.trend50 == 1).values
mB = (fe.day_move > -0.17).values
SG = [0.5, 1.0, 1.5, 2.0, 3.0]
TG = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, None]
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def simulate(evs, s, t):
    rows = []
    for i, L, side, a in evs:
        risk = s * a
        tgt = L + side * (t if t else 1e6) * risk
        r = sim_detail(i, L, side, L - side * risk, tgt)
        if r is None:
            continue
        j, px, why = r
        net = (px - L) * side - W.SP[i]
        rows.append((IDX[i], IDX[j], side, net, net / risk))
    T = pd.DataFrame(rows, columns=["進場時間", "出場時間", "dir", "usd", "R"])
    T["train"] = T.進場時間 < "2025-01-01"
    return T


def st(x):
    d = x.groupby(x.進場時間.dt.date).usd.sum()
    eq = d.cumsum()
    gp, gl = x.usd[x.usd > 0].sum(), -x.usd[x.usd < 0].sum()
    return dict(n=len(x) / x.進場時間.dt.date.nunique(), R=x.R.mean(), usd=x.usd.mean(), win=(x.usd > 0).mean(),
                pf=gp / gl, dd=(eq.cummax() - eq).max(), tot=x.usd.sum(), t=d.mean() / d.std() * np.sqrt(len(d)))


if __name__ == "__main__":
    p("第二十九部分：M15 突破 —— 過濾 A / A+B 下的停損與 RR 掃描")
    p("每格：訓練 R / 驗證 R；括號為全期每筆美元(每盎司)。t='收' = 不設停利、當天收盤出場\n")
    store = {}
    for fname, m in (("A", mA), ("A+B", mA & mB)):
        evs = [e for e, k in zip(ev, m) if k]
        for capn in (None, 1):
            p(f"=== 過濾 {fname}，{'疊單不限' if capn is None else '同方向一次 1 筆'} ===")
            p(f"{'':8s}" + "".join(f"{('t=' + (f'{t:g}' if t else '收')):>22s}" for t in TG))
            for s in SG:
                cells = []
                for t in TG:
                    T = cap(simulate(evs, s, t).sort_values("進場時間").reset_index(drop=True), capn)
                    a, b, c = st(T[T.train]), st(T[~T.train]), st(T)
                    store[(fname, capn, s, t)] = (a, b, c, T)
                    cells.append(f"{a['R']:+.3f}/{b['R']:+.3f}({c['usd']:+.2f})")
                p(f"s={s:<5}" + "".join(f"{x:>22s}" for x in cells))
            p()

    p("=== 細節：同方向一次 1 筆，各 RR（停損 2 ATR15）===")
    p(f"{'過濾':4s} {'t':>4s} {'筆/天':>6s} {'勝率':>6s} {'訓練R':>7s} {'驗證R':>7s} {'全期R':>7s} {'每筆$':>7s} {'PF':>5s} {'日t':>6s} {'總$':>8s} {'回撤$':>7s} | 逐年 R")
    for fname in ("A", "A+B"):
        for t in TG:
            a, b, c, T = store[(fname, 1, 2.0, t)]
            yr = " ".join(f"{T.R[T.進場時間.dt.year == y].mean():+.3f}" for y in (2023, 2024, 2025, 2026))
            p(f"{fname:4s} {(f'{t:g}' if t else '收'):>4s} {c['n']:6.1f} {c['win']:6.0%} {a['R']:+7.3f} {b['R']:+7.3f} {c['R']:+7.3f} "
              f"{c['usd']:+7.2f} {c['pf']:5.2f} {c['t']:+6.2f} {c['tot']:+8,.0f} {c['dd']:7,.0f} | {yr}")
    open("results_part29.txt", "w").write("\n".join(lines) + "\n")
