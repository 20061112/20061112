"""第三十部分：日線趨勢過濾（事先登記，看結果前決定）。
主規則：前一個交易日收盤 > 50 日均線 → 不做空；< 50 日均線 → 不做多（只用前一天以前的日線，沒有未來資料）。
鄰近：20 / 100 日均線、20 日漲跌方向。
對照：只做多（不看趨勢）、只做空、不加過濾。
套用在：版本 A（凍結門檻）、版本 A（滾動前推）、L1 核心 + 反向尾巴（前推）。
交易日 = broker 01:00 ~ 23:59（與 tpo.py 相同）。
"""
import numpy as np
import pandas as pd
from load import load
from part29_dir_filter import BASES, OOS_END, pf

out = open("results_part30.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


m1 = load("data/XAUUSD_M1_2023_2026.csv")
dayk = (m1.index - pd.Timedelta(hours=1)).normalize()
D = m1.groupby(dayk).agg(close=("close", "last"), n=("close", "size"))
D = D[D.n >= 600]
feat = pd.DataFrame(index=D.index)
for n in (20, 50, 100):
    feat[f"ma{n}"] = np.sign(D.close - D.close.rolling(n).mean())
feat["ret20"] = np.sign(D.close - D.close.shift(20))
feat = feat.shift(1)                                   # 只用前一天收盤以前


def trend_of(T, col):
    pos = np.searchsorted(feat.index.values, T.day.values, side="right") - 1
    return feat[col].to_numpy()[pos]


def apply(T, mode):
    s = T.side.to_numpy()
    if mode == "不加過濾":
        return T
    if mode == "只做多":
        return T[s == 1]
    if mode == "只做空":
        return T[s == -1]
    tr = trend_of(T, mode)
    return T[(tr == 0) | np.isnan(tr) | (tr == s)]


def stat(T):
    if len(T) == 0:
        return "n=0"
    d = T.groupby("day").R.sum()
    td = d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 else np.nan
    return f"{len(T):5d} 筆 PF {pf(T.pnl):4.2f}（R 計 {pf(T.R):4.2f}） 每筆 {T.R.mean():+.3f}R 累計 {T.R.sum():+7.1f}R 日t {td:+.1f}"


MODES = [("不加過濾", "不加過濾"), ("50 日均線（主規則）", "ma50"), ("20 日均線", "ma20"), ("100 日均線", "ma100"),
         ("20 日漲跌", "ret20"), ("只做多", "只做多"), ("只做空", "只做空")]
P("第三十部分：日線趨勢過濾（主規則 = 50 日均線：之上不做空、之下不做多）")
P("PF = 美元損益的獲利因子（毛利 / 毛損）；R 計 = 用 R 計算的 PF（不受金價高低影響）")
summary = []
for name, T in BASES.items():
    P(f"\n# {name}")
    for lab, mode in MODES:
        Y = apply(T, mode)
        o, i = Y[Y.day < OOS_END], Y[Y.day >= OOS_END]
        P(f"  {lab:14s} 全期 {stat(Y)}")
        P(f"  {'':14s} 樣本外 {stat(o)}")
        P(f"  {'':14s} 2026   {stat(i)}")
        yr = Y.groupby(Y.day.dt.year)
        P(f"  {'':14s} 逐年 " + "  ".join(f"{y}: {len(g)}筆 PF {pf(g.pnl):.2f} 多{(g.side == 1).sum()}/空{(g.side == -1).sum()}" for y, g in yr))
        summary.append(dict(base=name, filt=lab, n=len(Y), pf=pf(Y.pnl), n_o=len(o), pf_o=pf(o.pnl), r_o=o.R.mean(), n_i=len(i), pf_i=pf(i.pnl), r_i=i.R.mean()))

P("\n# 50 日均線主規則：每年有多少天在均線上 / 下（決定那年只做哪一邊）")
f50 = feat.ma50.dropna()
for y, g in f50.groupby(f50.index.year):
    P(f"  {y}: 之上 {(g > 0).mean():.0%} / 之下 {(g < 0).mean():.0%}")

S = pd.DataFrame(summary)
S.to_csv("part30_summary.csv", index=False, float_format="%.3f")
P("\n# 總表：筆數與 PF（全期 / 樣本外 2023/6~2025/12 / 2026）")
for _, r in S.iterrows():
    P(f"  {r.base:18s} {r.filt:14s} 全期 {r.n:5d} 筆 PF {r.pf:.2f} | 樣本外 {r.n_o:5d} 筆 PF {r.pf_o:.2f}（{r.r_o:+.3f}R）| 2026 {r.n_i:4d} 筆 PF {r.pf_i:.2f}（{r.r_i:+.3f}R）")
out.close()
