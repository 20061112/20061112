"""同一套策略搬到道瓊（DIA ↔ DDM 2 倍 / UDOW 3 倍），並與 QQQ、SOXX 在同一期間（2005/2 起）比較。
DDM / UDOW 與 DIA 追蹤同一個指數，可直接用真實價格（上市前用合成價格銜接）。
輸出 results_dow.txt、results_dow.csv。
"""
import numpy as np
import pandas as pd
import lev_lib as L

out = []
P = lambda s: (print(s, flush=True), out.append(s))

d0, q0, s0_ = L.build("dia"), L.build("qqq"), L.build("soxx")
common = d0.index.intersection(q0.index).intersection(s0_.index)      # 三者共同交易日（DIA 多一天 10/16）
d, q, s = d0.loc[common].copy(), q0.loc[common].copy(), s0_.loc[common].copy()
P(f"資料：DIA {d.index[0].date()} ~ {d.index[-1].date()}")
for t, k in [("ddm", 2), ("udow", 3)]:
    c = L.calibrate(d, t, k)
    P(f"校準 {t.upper()}（{k} 倍）：{c['start']} 起 真實年化 {c['real']:+.1%}、合成 {c['syn']:+.1%}、日報酬相關 {c['corr']:.3f}、追蹤誤差 {c['te']:.1%}")
d["ddm"] = L.real_series(d, "ddm", 2); d["udow"] = L.real_series(d, "udow", 3)
vol = {n: x.r.std() * np.sqrt(252) for n, x in [("DIA", d), ("QQQ", q), ("SOXX", s)]}
P("年化波動：" + "、".join(f"{n} {v:.0%}（3 倍耗損約 {3 * v ** 2:.0%}/年）" for n, v in vol.items()))

starts = list(range(2006, 2017))
rows = []
ad, zd, okd = L.signals(d); aq, zq, okq = L.signals(q); as_, zs, oks = L.signals(s)
lev_d = L.codes(d, okd); lev_q = L.codes(q, okq); lev_s = L.codes(s, oks)
dm_d = (~ad) & (d.close / d.close.shift(252) - 1 < d.rf.rolling(252).sum())

P("\n同期間（2005/2 起）")
R = {}
R["純定投 DIA"] = lambda a, b: L.dca_only(d, a, b)
R["純定投 QQQ"] = lambda a, b: L.dca_only(q, a, b)
R["DIA 策略（DIA ↔ 2 倍，真實 DDM）"] = L.strat(d, lev_d, ("base", "bill", "ddm"))
R["DIA 策略（DIA ↔ 3 倍，真實 UDOW）"] = L.strat(d, lev_d, ("base", "bill", "udow"))
R["DIA 策略 3 倍 + 雙動能"] = L.strat(d, L.codes(d, okd, dm_d), ("base", "bill", "udow"))
R["DIA 3 倍，用 QQQ 的訊號"] = L.strat(d, L.codes(d, okq.fillna(False)), ("base", "bill", "udow"))
R["QQQ 原策略（3 倍）"] = L.strat(q, lev_q, ("base", "bill", "x3"))
R["SOXX 策略（2 倍）"] = L.strat(s, lev_s, ("base", "bill", "x2"))
R["一半 QQQ 策略 + 一半 DIA 3 倍策略"] = L.mix([R["QQQ 原策略（3 倍）"], R["DIA 策略（DIA ↔ 3 倍，真實 UDOW）"]], [0.5, 0.5])
R["QQQ / DIA / SOXX 策略各 1/3"] = L.mix([R["QQQ 原策略（3 倍）"], R["DIA 策略（DIA ↔ 3 倍，真實 UDOW）"], R["SOXX 策略（2 倍）"]], [1 / 3] * 3)
for lab, fn in R.items():
    rows.append(L.evaluate(lab, fn, starts, P))
pd.DataFrame(rows).to_csv("results_dow.csv", index=False, float_format="%.4f")

# 日報酬相關（策略部位，一次投入）
def lump(x, lev, col3):
    held = np.r_[1, lev[:-1]]
    r1 = x.close.pct_change().fillna(0).values; r3 = x[col3].pct_change().fillna(0).values
    return pd.Series(np.where(held == 3, r3, r1), index=x.index)
C = pd.DataFrame(dict(QQQ=lump(q, lev_q, "x3"), DIA=lump(d, lev_d, "udow"), SOXX=lump(s, lev_s, "x2"))).dropna()
P("\n策略部位日報酬相關：\n" + C.corr().to_string(float_format=lambda x: f"{x:.2f}"))

P("\n區塊拼貼蒙地卡羅（DIA 歷史、3 個月區塊、300 條 10 年路徑）：年化 IRR")
M = L.mc(d)
P(f"  純定投 DIA：中位 {M.dca.median():.1%}、5% {M.dca.quantile(.05):.1%}、95% {M.dca.quantile(.95):.1%}")
for k in (2, 3):
    dd = M[f"s{k}"] - M.dca
    P(f"  DIA 策略 {k} 倍：中位 {M[f's{k}'].median():.1%}、5% {M[f's{k}'].quantile(.05):.1%}、95% {M[f's{k}'].quantile(.95):.1%}｜"
      f"勝過純定投 {(dd > 0).mean():.0%}、超額中位 {dd.median():+.1%}｜最慘相對本金 中位 {M[f's{k}w'].median():.0%}、5% {M[f's{k}w'].quantile(.05):.0%}")
dd = M.s3 - M.s2
P(f"  3 倍 − 2 倍：中位 {dd.median():+.1%}、3 倍較好的比例 {(dd > 0).mean():.0%}")
open("results_dow.txt", "w").write("\n".join(out) + "\n")
