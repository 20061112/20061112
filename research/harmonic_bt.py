"""諧波型態回測（不偷看未來）

進場：X-A-B-C 四個轉折都「已確認」後（AB/XA、BC/AB 合格），在理論 D 價掛限價單。
      D 價：加特利 0.786XA、蝙蝠 0.886XA、蝴蝶 1.27XA、螃蟹 1.618XA、鯊魚 0.886XC。
      D 價處的 CD/BC 也要合格。
      在 CD 這段結束（ZigZag 確認下一個轉折）之前沒碰到 D → 取消。
出場：TP1 = AD 的 0.382 回撤出半倉，TP2 = AD 的 0.618 回撤出另一半。
止損：加特利 1.0XA、蝙蝠 1.13XA、蝴蝶 1.618XA、螃蟹 1.902XA、鯊魚 1.272XC。
撮合用 M15 K 棒；同一根同時碰到止損和停利 → 算止損（保守）；進場那根只檢查止損。
成本：扣進場當根的點差。報酬以 R（= 進場到止損距離）計。

用法：python3 harmonic_bt.py [TF] [ZigZag ATR倍數]
"""
import sys
import numpy as np
import pandas as pd
from load import load
from harmonic import atr, inr, SPECS, TOL

ENTRY = {  # (基準, 進場比例, 止損比例)
    "加特利 Gartley":  ("xa", 0.786, 1.0),
    "蝙蝠 Bat":        ("xa", 0.886, 1.13),
    "蝴蝶 Butterfly":  ("xa", 1.27, 1.618),
    "螃蟹 Crab":       ("xa", 1.618, 1.902),
    "鯊魚 Shark":      ("xc", 0.886, 1.272),
}
TFLEN = {"M15": "15min", "H1": "1h", "H4": "4h"}


def zigzag_conf(df, k):
    """同 harmonic.zigzag，但多記錄每個轉折被確認的 bar。"""
    h, l, a = df["high"].values, df["low"].values, atr(df).bfill().values
    piv, d = [], 0
    ext_i, ext_p = 0, h[0]
    lo_i, lo_p, hi_i, hi_p = 0, l[0], 0, h[0]
    for i in range(1, len(df)):
        th = k * a[i]
        if d == 0:
            if h[i] > hi_p: hi_i, hi_p = i, h[i]
            if l[i] < lo_p: lo_i, lo_p = i, l[i]
            if hi_p - lo_p > th:
                if hi_i > lo_i:
                    piv.append((lo_i, lo_p, -1, i)); d, ext_i, ext_p = 1, hi_i, hi_p
                else:
                    piv.append((hi_i, hi_p, 1, i)); d, ext_i, ext_p = -1, lo_i, lo_p
        elif d == 1:
            if h[i] > ext_p: ext_i, ext_p = i, h[i]
            elif ext_p - l[i] > th:
                piv.append((ext_i, ext_p, 1, i)); d, ext_i, ext_p = -1, i, l[i]
        else:
            if l[i] < ext_p: ext_i, ext_p = i, l[i]
            elif h[i] - ext_p > th:
                piv.append((ext_i, ext_p, -1, i)); d, ext_i, ext_p = 1, i, h[i]
    return piv


def setups(df, k):
    piv = zigzag_conf(df, k)
    t = df.index
    out = []
    for j in range(3, len(piv) - 1):
        X, A, B, C = piv[j - 3:j + 1]
        nxt = piv[j + 1]
        xa, ab, bc = abs(A[1] - X[1]), abs(B[1] - A[1]), abs(C[1] - B[1])
        s = 1 if C[2] == 1 else -1   # C 高點 → D 往下 → 做多
        for name, (base, er, sr) in ENTRY.items():
            sp = SPECS[name]
            if not (inr(ab / xa, sp["ab"]) and inr(bc / ab, sp["bc"])):
                continue
            if base == "xa":
                D = A[1] - s * er * xa; SL = A[1] - s * sr * xa
            else:
                xc = abs(C[1] - X[1]); D = C[1] - s * er * xc; SL = C[1] - s * sr * xc
            if not inr(abs(C[1] - D) / bc, sp["cd"]):
                continue
            ad = abs(A[1] - D)
            out.append(dict(name=name, side=s, X=X[1], A=A[1], B=B[1], C=C[1], D=D, SL=SL,
                            TP1=D + s * 0.382 * ad, TP2=D + s * 0.618 * ad,
                            t_C=t[C[0]], t_active=t[C[3]], t_cancel=t[nxt[3]]))
    return out


def simulate(st, m15, tflen, be=False):
    o, h, l, sp = (m15[c].values for c in ("open", "high", "low", "spread"))
    idx = m15.index
    i0 = idx.searchsorted(st["t_active"] + tflen)
    i1 = idx.searchsorted(st["t_cancel"] + tflen)
    s, D, SL = st["side"], st["D"], st["SL"]
    fill = None
    for i in range(i0, min(i1, len(idx))):
        if (s == 1 and l[i] <= D) or (s == -1 and h[i] >= D):
            fill = i
            break
    if fill is None:
        return None
    ent = min(o[fill], D) if s == 1 else max(o[fill], D)
    if s * (ent - SL) <= 0:          # 跳空直接越過止損
        ent = o[fill]
    risk = abs(ent - SL)
    cost = sp[fill] * 0.01
    res = dict(t_entry=idx[fill], entry=ent, risk=risk)
    stop, half_done, pnl = SL, False, 0.0
    i = fill
    # 進場那根：只看止損
    if (s == 1 and l[i] <= SL) or (s == -1 and h[i] >= SL):
        res.update(t_exit=idx[i], pnl=s * (SL - ent) - cost, outcome="止損")
        return res
    for i in range(fill + 1, len(idx)):
        hit_sl = (s == 1 and l[i] <= stop) or (s == -1 and h[i] >= stop)
        if hit_sl:
            px = o[i] if s * (o[i] - stop) < 0 else stop   # 跳空
            q = 0.5 if half_done else 1.0
            pnl += q * s * (px - ent)
            res.update(t_exit=idx[i], pnl=pnl - cost,
                       outcome="TP1後保本" if half_done and be else ("TP1後止損" if half_done else "止損"))
            return res
        if not half_done and s * (h[i] if s == 1 else l[i]) >= s * st["TP1"]:
            pnl += 0.5 * s * (st["TP1"] - ent); half_done = True
            if be: stop = ent
        if half_done and s * (h[i] if s == 1 else l[i]) >= s * st["TP2"]:
            pnl += 0.5 * s * (st["TP2"] - ent)
            res.update(t_exit=idx[i], pnl=pnl - cost, outcome="TP1+TP2")
            return res
    res.update(t_exit=idx[-1], pnl=pnl + (0.5 if half_done else 1) * s * (m15["close"].iloc[-1] - ent) - cost,
               outcome="未結")
    return res


def summary(tr):
    g = tr.groupby("name")
    rows = []
    for name, x in list(g) + [("全部", tr)]:
        R = x["R"]
        oc = x["outcome"].value_counts()
        rows.append(dict(型態=name, 筆數=len(x), 勝率=(x["pnl"] > 0).mean(),
                         TP2全中=oc.get("TP1+TP2", 0) / len(x),
                         TP1後回止損=(oc.get("TP1後止損", 0) + oc.get("TP1後保本", 0)) / len(x),
                         直接止損=oc.get("止損", 0) / len(x),
                         平均R=R.mean(), t值=R.mean() / (R.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                         總R=R.sum(), 平均賺R=R[R > 0].mean(), 平均賠R=R[R <= 0].mean(),
                         損益美元=x["pnl"].sum()))
    return pd.DataFrame(rows)


def run(tf, k, be=False, quiet=False):
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    df = m15 if tf == "M15" else m15.resample(TFLEN[tf]).agg(
        dict(open="first", high="max", low="min", close="last", tickvol="sum", spread="mean")).dropna()
    tflen = pd.Timedelta(TFLEN[tf])
    trades = []
    for st in setups(df, k):
        r = simulate(st, m15, tflen, be)
        if r:
            trades.append({**st, **r})
    tr = pd.DataFrame(trades)
    tr["R"] = tr["pnl"] / tr["risk"]
    tr["year"] = tr["t_entry"].dt.year
    return tr


def main():
    tf = sys.argv[1] if len(sys.argv) > 1 else "H1"
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)
    for be in (False, True):
        tr = run(tf, k, be)
        print(f"=== {tf} ZigZag {k}×ATR  2023/01~2026/10  "
              f"{'TP1 後止損移到保本' if be else 'TP1 後止損不動（你的規則）'} ===")
        print(summary(tr).to_string(index=False))
        if not be:
            print("\n分年度（平均R / 筆數）：")
            print(tr.pivot_table(index="name", columns="year", values="R",
                                 aggfunc=lambda v: f"{v.mean():+.2f} / {len(v)}").to_string())
            cols = ["name", "side", "t_entry", "entry", "SL", "TP1", "TP2", "outcome", "t_exit", "pnl", "R"]
            tr[cols].to_csv(f"harmonic_trades_{tf}.csv", index=False)
            print("\n最近 10 筆：")
            print(tr[cols].tail(10).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
