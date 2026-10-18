"""第二十二部分：版本 A 換商品的樣本外測試 —— NAS100 M1（2025/10/1 ~ 2026/10/16）。

規則、時段（broker 02:00~10:30 起的 60 分 IB）、篩選分位數一律沿用版本 A，一個都沒改。
NAS100 價格 ÷ 5（tpo.INSTR），讓 TPO 格子相對 ATR 的解析度和黃金一樣；R 不受影響，點數 = 換算後 × 5。
兩種門檻：
  (1) 滾動前推：每月月初用 NAS100 自己過去的訊號重算分位數（= 版本 A 的實盤做法）
  (2) 完全凍結：直接套用黃金的 final_A_live.json 門檻（都是 ATR 比例，可直接搬）
輸出：results_part22.txt、part22_nas100_trades.csv、part22_nas100_equity.png
"""
import json
import os
os.environ["SYMBOL"] = "NAS100"
import numpy as np
import pandas as pd
from tpo import INSTR
from part14_deep import ARR, D, DAYS
from part17_balance import gen, wf, score

K = INSTR["NAS100"][1]
CUT = pd.Timestamp("2026-05-01")          # 前後兩半（各約 5 個月）
out = open("results_part22.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def pf(p):
    return p[p > 0].sum() / max(-p[p <= 0].sum(), 1e-9)


def line(Y, lab):
    if not len(Y):
        return f"{lab:36s} 0 筆"
    a, b = Y[Y.day < CUT], Y[Y.day >= CUT]
    s = score(Y)
    r = Y.R
    t = r.mean() / r.std() * np.sqrt(len(r))
    return (f"{lab:36s} {len(Y):4d}筆 {s['per_day']:.1f}/日 每筆 {r.mean():+.3f}R t={t:+.2f} 勝率 {s['win']:.0f}% PF {s['pf']:.2f} "
            f"總 {r.sum():+6.1f}R  回撤 {s['dd'] * K:.0f}點 Sharpe {s['sharpe']:+.2f} 月PF>1 {s['mon_ok']:.0f}% "
            f"| 前半 {a.R.mean():+.3f}R PF {pf(a.pnl):.2f} | 後半 {b.R.mean():+.3f}R PF {pf(b.pnl):.2f}")


def er_day(r):
    """同 part21_er.er_feats 的 er_day（那兩個檔案 import 時會直接跑整段研究，所以複製過來）。"""
    t, O, H, L, C, S = ARR[r.day]
    c = C[:r.i]
    return abs(c[-1] - O[0]) / np.abs(np.diff(c)).sum() if len(c) >= 30 else np.nan


def wf_er(X, qt, qe, q_er=0.8, q_ib=0.4):
    """同 part21_simple.wf_er。"""
    X = X.copy(); X["ym"] = X.day.dt.to_period("M"); parts = []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        k = ~(cur.aligned & (cur.ib_atr < past.ib_atr.quantile(q_ib))) & (cur["tail"] <= past["tail"].quantile(qt)) \
            & (cur.ext_letter.fillna(0) <= past.ext_letter.quantile(qe)) & (cur.er_day.fillna(0) <= past.er_day.quantile(q_er))
        parts.append(cur[k])
    return pd.concat(parts).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])


def frozen(X, live):
    Y = X[~(X.aligned & (X.ib_atr < live["thr_ib_atr"])) & (X["tail"] <= live["thr_tail_atr"])
          & (X.ext_letter.fillna(0) <= live["thr_ext_letter"])]
    Y = Y[Y.day.dt.to_period("M") >= X.day.dt.to_period("M").min() + 2]     # 和滾動前推同一段期間
    return Y.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])


if __name__ == "__main__":
    live = json.load(open("final_A_live.json"))
    X = gen("close5", 0.05, (2, 10), 30, False)
    A = wf(X, 0.9, 0.9).sort_values("t_in").reset_index(drop=True)
    P(f"第二十二部分：版本 A 在 NAS100 的樣本外測試（{DAYS[0].date()} ~ {DAYS[-1].date()}，{len(DAYS)} 個交易日）")
    P(f"報告期間 {A.day.min().date()} ~ {A.day.max().date()}（前 2 個月只用來算門檻）；前半 < {CUT.date()} ≤ 後半")
    P(f"ATR10 中位數 {X.atr.median() * K:.0f} 指數點；停損距離中位數 {A.risk.median() * K:.0f} 點（{(A.risk / A.atr).median():.2f} ATR）；"
      f"點差中位數 {A.spread.median() * K:.2f} 點（{(A.spread / A.risk).median() * 100:.1f}% R）")
    P("黃金版本 A 參考（2026/3~10 滾動前推）：735 筆 6.1/日 每筆 +0.46R 勝率 47.5% PF 1.92 Sharpe 4.62")

    P("\n## 1. 版本 A")
    P("  " + line(A, "滾動前推（NAS100 自己的門檻）"))
    P("  " + line(frozen(X, live), "完全凍結（黃金門檻）"))

    P("\n## 2. 複雜度階梯（每一步只多一條規則，滾動前推）")
    steps = [("L0 觸價、停損另一側、收盤出", ("touch", 0.0, True), (1.0, 1.0, 0.0)),
             ("L1 + M5 收盤 + 0.05 ATR", ("close5", 0.05, True), (1.0, 1.0, 0.0)),
             ("L2 + 不做星期一", ("close5", 0.05, False), (1.0, 1.0, 0.0)),
             ("L3 + 順前日方向的小 IB 不做", ("close5", 0.05, False), (1.0, 1.0, 0.4)),
             ("L4 + 反向尾巴", ("close5", 0.05, False), (0.9, 1.0, 0.4)),
             ("L5 + 反向極值字母（= A）", ("close5", 0.05, False), (0.9, 0.9, 0.4))]
    for lab, (cf, bf, mon), (qt, qe, qi) in steps:
        Y = wf(gen(cf, bf, (2, 10), 30, mon), qt, qe, qi)
        P("  " + line(Y, lab))
        r = Y.R.sort_values(ascending=False)
        P(f"  {'':36s} 拿掉最好 10 / 20 筆後總 R：{r.iloc[10:].sum():+.1f} / {r.iloc[20:].sum():+.1f}")

    P("\n## 3. 第二十一部分的 ER 變體")
    XE = X.assign(er_day=[er_day(r) for r in X.itertuples()])
    P("  " + line(wf_er(XE, 0.9, 0.9), "A + ER"))
    P("  " + line(wf_er(XE, 0.9, 1.0), "tail + ER"))

    P("\n## 4. 版本 A 分組（R）")
    def grp(g):
        a = A.groupby(g).agg(筆數=("R", "size"), 勝率=("R", lambda x: (x > 0).mean() * 100), 每筆R=("R", "mean"), 總R=("R", "sum"))
        a["PF"] = A.groupby(g).pnl.apply(pf)
        return a.round(2).to_string()
    A["ib_start"] = A.h.map(lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}")
    P("[月份]\n" + grp(A.day.dt.strftime("%Y-%m")))
    P("\n[方向]\n" + grp(A.side.map({1: "多", -1: "空"})))
    P("\n[IB 起點]\n" + grp("ib_start"))
    P("\n[出場原因]\n" + grp("exit_reason"))
    hb = pd.cut(A.hold_h, [0, 2, 8, 12, 16, 24], right=False)
    P("\n[持倉時間]\n" + grp(hb))

    P("\n## 5. 方向拆開（對照同期大盤漂移）")
    P("  " + line(A[A.side == 1], "版本 A 只看多單"))
    P("  " + line(A[A.side == -1], "版本 A 只看空單"))
    d0, d1 = A.day.min(), A.day.max()
    P(f"  同期 NAS100 {D.loc[d0].open * K:.0f} → {D.loc[d1].close * K:.0f}（{(D.loc[d1].close / D.loc[d0].open - 1) * 100:+.1f}%）；"
      f"上漲日 {(D.loc[d0:d1].close > D.loc[d0:d1].open).mean() * 100:.0f}%")

    A.assign(**{c: A[c] * K for c in ["lvl", "ibh", "ibl", "risk", "pnl", "mae", "mfe", "spread", "atr"]}) \
        [["day", "side", "ib_start", "ibh", "ibl", "t_in", "lvl", "risk", "exit_time", "exit_px", "exit_reason", "pnl", "R",
          "mae", "mfe", "hold_h", "atr", "ib_atr", "aligned", "tail", "ext_letter", "spread"]] \
        .assign(exit_px=lambda d: d.exit_px * K) \
        .to_csv("part22_nas100_trades.csv", index=False, float_format="%.3f")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for Y, lab, c in [(A, "Version A (walk-forward thresholds)", "#3b6ea8"), (frozen(X, live), "Version A (gold thresholds, frozen)", "#c9824a"),
                      (wf(gen("touch", 0.0, (2, 10), 30, True), 1.0, 1.0, 0.0), "L0 core (no filters)", "#888888")]:
        Y = Y.sort_values("exit_time")
        ax.plot(Y.exit_time, Y.R.cumsum(), lw=1.2, color=c, label=lab, drawstyle="steps-post")
    ax.axhline(0, color="k", lw=.6); ax.set_ylabel("cumulative R"); ax.legend(loc="upper left")
    ax.set_title("NAS100 M1 — Version A rules unchanged (out-of-sample instrument)")
    ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("part22_nas100_equity.png", dpi=110)
    out.close()
