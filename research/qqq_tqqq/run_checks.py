"""檢查 1：震盪年（2005、2011、2015）與強勢年各變體的逐年報酬（一次投入、不定投）。
檢查 2：3 個月區塊拼貼蒙地卡羅（與原報告同法），比較 BASE 與趨勢集成 ENS_T、MA 鄰域。
輸出 results_checks.txt。
"""
import numpy as np
import pandas as pd
import indicators as ind
import engine as E
from run_tests import q, OK0, ZOK, c, r, w_rule, w_cont

out = []
P = lambda s: (print(s), out.append(s))

# ---------------- 檢查 1
tr_votes = lambda cc: sum([(cc > cc.rolling(m).mean()).astype(float) for m in [150, 200, 250]] +
                          [(ind.tsmom(cc, n) > 0).astype(float) for n in [126, 189, 252]]) / 6
V = {
    "BASE": w_rule(OK0),
    "ENS_T 趨勢集成": w_cont(tr_votes(c) * ZOK),
    "ER60>0.1 AND": w_rule(OK0 & (ind.er(c, 60) > 0.1).fillna(True)),
    "排列熵60 z<0 AND": w_rule(OK0 & (ind.rz(ind.perm_entropy(c, 3, 60)) < 0).fillna(True)),
    "VarRatio5/250>1 AND": w_rule(OK0 & (ind.var_ratio(r, 5, 250) > 1).fillna(True)),
    "CUSUM h6 AND": w_rule(OK0 & (ind.cusum_state(r, 0.25, 6) > 0)),
    "VXN z<0.5 AND": w_rule(OK0 & (ind.rz(np.log(q.vxn)) < 0.5).fillna(True)),
    "HAR σ<25% AND": w_rule(OK0 & (ind.har_vol(q) < 0.25).fillna(True)),
}
yrs = [2005, 2011, 2015, 2018, 2022, 2009, 2013, 2023]
rows = {}
for k, w in V.items():
    cal, _ = E.calendar_returns(q, w)
    rows[k] = cal.reindex(yrs)
rows["QQQ"] = (1 + q.r).groupby(q.index.year).prod().reindex(yrs) - 1
T = pd.DataFrame(rows).T
P("逐年報酬（一次投入）：")
P(T.to_string(float_format=lambda x: f"{x:+.0%}"))

# ---------------- 檢查 2：區塊拼貼
rng = np.random.default_rng(7)
base = q[["open", "high", "low", "close"]].copy()
lr = np.log(base.close).diff().fillna(0).values
ho = np.log(base.high / base.close).values; lo = np.log(base.low / base.close).values
oo = np.log(base.open / base.close).values
rf = q.rf.values
B = 63; N = 2520 + 300  # 300 天暖身 + 10 年
idx = pd.bdate_range("2000-01-03", periods=N)


def path():
    pos = []
    while len(pos) < N:
        s = rng.integers(1, len(lr) - B)
        pos.extend(range(s, s + B))
    pos = np.array(pos[:N])
    cl = 100 * np.exp(np.cumsum(lr[pos]))
    d = pd.DataFrame(dict(close=cl, high=cl * np.exp(ho[pos]), low=cl * np.exp(lo[pos]),
                          open=cl * np.exp(oo[pos])), index=idx)
    d["r"] = d.close.pct_change().fillna(0)
    d["rf"] = rf[pos]
    r3 = 3 * d.r - E.FEE / 252 - 2 * (d.rf * 252 + E.SPREAD) / 252
    d["tq_syn"] = (1 + r3).cumprod()
    p = d.index.to_period("W").to_series(index=d.index)
    d["week_start"] = p.ne(p.shift(1)).values
    return d


def wr(qq, ok):
    return E.weights_from_rules(qq, ok, ~ok)


def wc(qq, t):
    t = pd.Series(t, index=qq.index).shift(1).fillna(0).clip(0, 1).values
    ws = qq.week_start.values; w = np.zeros(len(t)); cur = 0.0
    for i in range(len(t)):
        cur = t[i] if (t[i] < cur or ws[i]) else cur
        w[i] = cur
    return w


res = []
for k in range(300):
    d = path()
    z = E.atr_z(d); zok = z < 0.5; cc = d.close
    ws = {"BASE": wr(d, (cc > cc.rolling(200).mean()) & zok),
          "ENS_T": wc(d, tr_votes(cc) * zok),
          "MA250": wr(d, (cc > cc.rolling(250).mean()) & zok),
          "TSMOM220": wr(d, (ind.tsmom(cc, 220) > 0) & zok)}
    row = {}
    s0 = d.index[300]
    for name, w in ws.items():
        row[name] = E.xirr(E.run(d, w, start=s0, z=z)["flows"])
    row["DCA"] = E.xirr(E.run(d, ws["BASE"], start=s0, dca_only=True)["flows"])
    res.append(row)
R = pd.DataFrame(res)
P("\n區塊拼貼蒙地卡羅（3 個月區塊、10 年、300 條）：年化 IRR")
P(R.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "5%", "50%", "95%"]].to_string(float_format=lambda x: f"{x:.3f}"))
for k in ["ENS_T", "MA250", "TSMOM220"]:
    d = R[k] - R.BASE
    P(f"{k} − BASE：中位 {d.median():+.4f}、平均 {d.mean():+.4f}、勝率 {(d > 0).mean():.0%}、5% {d.quantile(.05):+.4f}、95% {d.quantile(.95):+.4f}")
P(f"BASE 勝過 DCA：{(R.BASE > R.DCA).mean():.0%}，超額中位 {(R.BASE - R.DCA).median():+.4f}")
open("results_checks.txt", "w").write("\n".join(out) + "\n")
