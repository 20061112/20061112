"""同一套策略搬到標普 500（SPY ↔ SSO 2 倍 / UPRO 3 倍），並與 QQQ、DIA、SOXX 在共同期間比較。
SSO / UPRO 與 SPY 追蹤同一個指數，上市後用真實價格、之前用合成銜接。
輸出 results_spy.txt、results_spy.csv。
"""
import numpy as np
import pandas as pd
import lev_lib as L

out = []
P = lambda s: (print(s, flush=True), out.append(s))

raw = {t: L.build(t) for t in ["spy", "qqq", "dia", "soxx"]}
common = raw["spy"].index
for t in raw:
    common = common.intersection(raw[t].index)
X = {t: raw[t].loc[common].copy() for t in raw}
sp, q, d, s = X["spy"], X["qqq"], X["dia"], X["soxx"]
P(f"共同期間：{common[0].date()} ~ {common[-1].date()}")
for t, k in [("sso", 2), ("upro", 3)]:
    c = L.calibrate(sp, t, k)
    P(f"校準 {t.upper()}（{k} 倍）：{c['start']} 起 真實年化 {c['real']:+.1%}、合成 {c['syn']:+.1%}、日報酬相關 {c['corr']:.3f}、追蹤誤差 {c['te']:.1%}")
sp["sso"] = L.real_series(sp, "sso", 2); sp["upro"] = L.real_series(sp, "upro", 3)
d["udow"] = L.real_series(d, "udow", 3)
v = sp.r.std() * np.sqrt(252)
P(f"SPY 年化波動 {v:.0%}（3 倍耗損約 {3 * v ** 2:.0%}/年）")

starts = list(range(2006, 2017))
a_sp, z_sp, ok_sp = L.signals(sp)
lev_sp = L.codes(sp, ok_sp)
dm_sp = (~a_sp) & (sp.close / sp.close.shift(252) - 1 < sp.rf.rolling(252).sum())
lev_q = L.codes(q, L.signals(q)[2]); lev_d = L.codes(d, L.signals(d)[2]); lev_s = L.codes(s, L.signals(s)[2])

R = {}
R["純定投 SPY"] = lambda a, b: L.dca_only(sp, a, b)
R["SPY 策略 2 倍（真實 SSO）"] = L.strat(sp, lev_sp, ("base", "bill", "sso"))
R["SPY 策略 3 倍（真實 UPRO）"] = L.strat(sp, lev_sp, ("base", "bill", "upro"))
R["SPY 策略 3 倍 + 雙動能"] = L.strat(sp, L.codes(sp, ok_sp, dm_sp), ("base", "bill", "upro"))
R["SPY 3 倍，用 QQQ 的訊號"] = L.strat(sp, L.codes(sp, L.signals(q)[2]), ("base", "bill", "upro"))
R["純定投 QQQ"] = lambda a, b: L.dca_only(q, a, b)
R["QQQ 原策略（3 倍）"] = L.strat(q, lev_q, ("base", "bill", "x3"))
R["一半 QQQ 策略 + 一半 SPY 3 倍策略"] = L.mix([R["QQQ 原策略（3 倍）"], R["SPY 策略 3 倍（真實 UPRO）"]], [0.5, 0.5])
P("\n同期間")
rows = [L.evaluate(lab, fn, starts, P) for lab, fn in R.items()]
pd.DataFrame(rows).to_csv("results_spy.csv", index=False, float_format="%.4f")

# 四個指數總表（各自最適合的槓桿）
P("\n四個指數總表（同期間；DIA、SPY 用真實 3 倍，SOXX 用合成 2 倍，QQQ 用合成 3 倍）")
T = {
    "QQQ 純定投": lambda a, b: L.dca_only(q, a, b),
    "QQQ 策略 3 倍": R["QQQ 原策略（3 倍）"],
    "SPY 純定投": R["純定投 SPY"],
    "SPY 策略 3 倍": R["SPY 策略 3 倍（真實 UPRO）"],
    "DIA 純定投": lambda a, b: L.dca_only(d, a, b),
    "DIA 策略 3 倍": L.strat(d, lev_d, ("base", "bill", "udow")),
    "SOXX 純定投": lambda a, b: L.dca_only(s, a, b),
    "SOXX 策略 2 倍": L.strat(s, lev_s, ("base", "bill", "x2")),
}
for lab, fn in T.items():
    L.evaluate(lab, fn, starts, P)


def lump(x, lev, col3):
    held = np.r_[1, lev[:-1]]
    return pd.Series(np.where(held == 3, x[col3].pct_change().fillna(0).values, x.close.pct_change().fillna(0).values), index=x.index)


C = pd.DataFrame(dict(QQQ=lump(q, lev_q, "x3"), SPY=lump(sp, lev_sp, "upro"), DIA=lump(d, lev_d, "udow"), SOXX=lump(s, lev_s, "x2")))
P("\n策略部位日報酬相關：\n" + C.corr().to_string(float_format=lambda x: f"{x:.2f}"))

P("\n區塊拼貼蒙地卡羅（SPY 歷史、3 個月區塊、300 條 10 年路徑，合成、不含股息）")
M = L.mc(sp)
P(f"  純定投 SPY：中位 {M.dca.median():.1%}、5% {M.dca.quantile(.05):.1%}、95% {M.dca.quantile(.95):.1%}")
for k in (2, 3):
    dd = M[f"s{k}"] - M.dca
    P(f"  SPY 策略 {k} 倍：中位 {M[f's{k}'].median():.1%}、5% {M[f's{k}'].quantile(.05):.1%}、95% {M[f's{k}'].quantile(.95):.1%}｜"
      f"勝過純定投 {(dd > 0).mean():.0%}、超額中位 {dd.median():+.1%}｜最慘相對本金 中位 {M[f's{k}w'].median():.0%}、5% {M[f's{k}w'].quantile(.05):.0%}")
dd = M.s3 - M.s2
P(f"  3 倍 − 2 倍：中位 {dd.median():+.1%}、3 倍較好的比例 {(dd > 0).mean():.0%}")
open("results_spy.txt", "w").write("\n".join(out) + "\n")

# 四個指數用同一段（2005 起）歷史、同一套方法的蒙地卡羅
P("\n四個指數的區塊拼貼蒙地卡羅（皆用 2005 起歷史、300 條 10 年路徑、合成槓桿、不含股息）")
P("  指數    純定投中位  策略 2 倍中位（勝率、超額）   策略 3 倍中位（勝率、超額）   3 倍最慘 5%")
for t, x in [("QQQ", q), ("SPY", sp), ("DIA", d), ("SOXX", s)]:
    M = L.mc(x)
    d2 = M.s2 - M.dca; d3 = M.s3 - M.dca
    P(f"  {t:5s}  {M.dca.median():8.1%}   {M.s2.median():8.1%}（{(d2 > 0).mean():.0%}、{d2.median():+.1%}）   "
      f"{M.s3.median():8.1%}（{(d3 > 0).mean():.0%}、{d3.median():+.1%}）   {M.s3w.quantile(.05):.0%}")
open("results_spy.txt", "w").write("\n".join(out) + "\n")
