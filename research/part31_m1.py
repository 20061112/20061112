"""第三十一部分：M1 版 OW 深入研究。

A. 拆解 M1 vs M5（2026/1~10/8）：同一筆交易（同日、同 IB、同方向）的 R 差 vs 只有 M1 / 只有 M5 有的交易
B. 執行現實：延遲一根才進場（下一根 M1 開盤）+ 額外滑價
C. M1 參數穩健：緩衝、連續兩根 M1 收盤確認、突破窗、起點間隔
D. 多年代理：「碰到 IB 邊 + 緩衝就進場」（最早確認，M1 收盤介於它和 M5 收盤之間）vs 收盤確認，用 M15 2023~2026、M5 2025~2026
E. M1 的逐月、時段穩定性；假突破（進場後 15 分鐘內收回 IB 內）比例
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, prev_profiles, label_ow, PER, pf

out = open("results_part31.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
M1, M5, M15 = ("data/XAUUSD_M1_2026.csv", 1), ("data/XAUUSD_M5_2025_2026.csv", 5), ("data/XAUUSD_M15_2023_2026.csv", 15)
A0, B0 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-08")
GAP = ("2026-04-03", "2026-04-14")
_PREV = {}


def ow(src, latency=False, consec=1, **kw):
    path, bm = src
    days, D, ARR = data(path, bm)
    S = signals(days, D, ARR, bm, monday=True, **kw)
    if consec == 2 and len(S):                      # 需要連續兩根收盤都在外：確認延到下一根（若下一根也在外）
        keep, rows = [], []
        for r in S.itertuples():
            t, O, H, L, C, Sp, mins = ARR[r.day]; k = r.i         # r.i = 確認根的下一根
            if k < len(C) and ((C[k] > r.lvl - 1e-9 and r.side == 1 and C[k] > r.ibh) or (r.side == -1 and C[k] < r.ibl)):
                d = r._asdict(); d.update(i=k + 1, lvl=C[k], t_in=t[k] + pd.Timedelta(minutes=bm)); rows.append(d)
        S = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    if latency and len(S):
        rows = []
        for r in S.itertuples():
            t, O, H, L, C, Sp, mins = ARR[r.day]
            if r.i < len(O):
                d = r._asdict(); d.update(lvl=O[r.i]); rows.append(d)
        S = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    X = simulate(S, ARR, bm)
    if path not in _PREV:
        _PREV[path] = prev_profiles(days, ARR)
    X["ow"] = label_ow(X, _PREV[path])
    X = X[X.ow].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    X["key"] = X.day.dt.strftime("%Y-%m-%d") + "|" + X.h.round(3).astype(str) + "|" + X.side.astype(str)
    return X, days


def in26(X):
    return X[(X.day >= A0) & (X.day <= B0) & ~X.day.between(*GAP)]


def s(X):
    if not len(X):
        return "0 筆"
    return f"{len(X):4d}筆 勝率 {(X.pnl > 0).mean() * 100:3.0f}% 每筆 {X.R.mean():+.3f}R PF {pf(X.R):.2f} 總 {X.R.sum():+5.0f}R"


# ---------- A ----------
X1, d1 = ow(M1); X5, d5 = ow(M5); X15, d15 = ow(M15)
a1, a5 = in26(X1), in26(X5)
P("第三十一部分 A：M1 比 M5 好在哪？（2026/1~10/8）")
P(f"  M1 全部 {s(a1)}\n  M5 全部 {s(a5)}")
both = set(a1.key) & set(a5.key)
m1b, m5b = a1[a1.key.isin(both)].set_index("key"), a5[a5.key.isin(both)].set_index("key")
j = m1b.join(m5b, lsuffix="_1", rsuffix="_5")
P(f"  同一筆交易（{len(j)} 筆）：M1 每筆 {j.R_1.mean():+.3f}R vs M5 {j.R_5.mean():+.3f}R；M1 早進場 {((j.t_in_5 - j.t_in_1).dt.total_seconds() / 60).median():.0f} 分鐘（中位），"
      f"進場價較有利 {(j.side_1 * (j.lvl_5 - j.lvl_1)).median():+.2f} 美元（中位）；同一筆 M1 勝、M5 敗 {((j.pnl_1 > 0) & (j.pnl_5 <= 0)).sum()} 筆 / 反之 {((j.pnl_1 <= 0) & (j.pnl_5 > 0)).sum()} 筆")
P(f"  只有 M1 有的交易：{s(a1[~a1.key.isin(both)])}")
P(f"  只有 M5 有的交易：{s(a5[~a5.key.isin(both)])}")
P(f"  → M1 多出的總 R {a1.R.sum() - a5.R.sum():+.0f}：同一筆交易貢獻 {(j.R_1 - j.R_5).sum():+.0f}，交易組成差異貢獻 "
  f"{a1[~a1.key.isin(both)].R.sum() - a5[~a5.key.isin(both)].R.sum():+.0f}")

# ---------- B ----------
P("\n第三十一部分 B：執行現實（M1，2026）")
L1, _ = ow(M1, latency=True); l1 = in26(L1)
for lab, X in [("M1 收盤價進場（回測）", a1), ("M1 延遲一根：下一根開盤進場", l1)]:
    cells = []
    for c in (0, 0.3, 0.5, 1.0):
        r = (X.R * X.risk - c) / X.risk
        cells.append(f"+{c}美元: {r.mean():+.3f}R/PF{pf(r):.2f}")
    P(f"  {lab:28s} " + "  ".join(cells))
r5 = a5.R
P(f"  （對照 M5 收盤進場、不加滑價：{r5.mean():+.3f}R/PF{pf(r5):.2f}；M5 延遲一根）" + "  ".join(
    f"+{c}: {((in26(ow(M5, latency=True)[0]).R * in26(ow(M5, latency=True)[0]).risk - c) / in26(ow(M5, latency=True)[0]).risk).mean():+.3f}R" for c in (0, 0.5)))

# ---------- C ----------
P("\n第三十一部分 C：M1 參數穩健（2026；每格 = 筆數 / 每筆 R / PF）")
for lab, kw in [("基準（緩衝 0.05ATR、窗 3h、每 30 分）", {}), ("緩衝 0", dict(buf=0.0)), ("緩衝 0.02", dict(buf=0.02)),
                ("緩衝 0.1", dict(buf=0.10)), ("連續兩根 M1 收盤確認", dict(consec=2)), ("突破窗 2h", dict(win=2)),
                ("突破窗 5h", dict(win=5)), ("起點每 60 分", dict(step=60)), ("IB 30 分", dict(ib_min=30)), ("IB 90 分", dict(ib_min=90))]:
    X, _ = ow(M1, **kw)
    P(f"  {lab:34s} {s(in26(X))}")

# ---------- D ----------
P("\n第三十一部分 D：多年代理 ——「碰到就進場」（最早確認）vs 收盤確認（每格 = 每筆 R / PF / 筆數）")
for lab, src, kw in [("M15 收盤確認", M15, {}), ("M15 碰到 IB邊+緩衝 就進場", M15, dict(confirm="touch")),
                     ("M5 收盤確認", M5, {}), ("M5 碰到 IB邊+緩衝 就進場", M5, dict(confirm="touch"))]:
    X, _ = ow(src, **kw)
    cells = [f"{n} {x.R.mean():+.2f}/PF{pf(x.R):.2f}/{len(x)}" for n, a, b in PER for x in [X[(X.day >= a) & (X.day <= b)]] if len(x)]
    P(f"  {lab:26s} " + "  ".join(cells))

# ---------- E ----------
P("\n第三十一部分 E：M1 穩定性（2026）")
P("  逐月：" + "  ".join(f"{m}月 {g.R.sum():+.0f}R/{len(g)}筆" for m, g in a1.groupby(a1.day.dt.month)))
P("  前半（1~5 月）" + s(a1[a1.day < "2026-06-01"]) + " | 後半（6~10 月）" + s(a1[a1.day >= "2026-06-01"]))
hb = pd.cut(a1.h, [1.9, 4, 6, 8, 11], labels=["02~04", "04~06", "06~08", "08~11"])
P("  IB 起點：" + "  ".join(f"{k} {g.R.mean():+.2f}R/PF{pf(g.R):.2f}({len(g)})" for k, g in a1.groupby(hb, observed=True)))
days, D, ARR = data(*M1)
fb = []
for r in a1.itertuples():
    t, O, H, L, C, Sp, mins = ARR[r.day]
    seg = C[r.i:r.i + 15]
    fb.append(bool(((seg < r.ibh) if r.side == 1 else (seg > r.ibl)).any()))
a1 = a1.assign(fakeback=fb)
P(f"  進場後 15 分鐘內收回 IB 內（假突破跡象）：{np.mean(fb) * 100:.0f}% 的單；這些單 {s(a1[a1.fakeback])}；其餘 {s(a1[~a1.fakeback])}")
a1.to_csv("part31_m1_trades.csv", index=False)
out.close()
