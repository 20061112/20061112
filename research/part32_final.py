"""第三十二部分：定案 H1 六合一（3R 停利）的逐筆明細、統計與美元換算。
規則同第二十八部分，唯一差別：停利 2R → 3R（第三十部分只用研究期選出；第三十一部分最大停損 / 時間停損 / 虧損上限都沒有在研究期勝過它）。
輸出：part32_final_trades.csv、results_part32_final.txt、part32_final_equity.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import part28_h1_combo as P
from part22_divergence import get_frames, nonoverlap, SPLIT
from part30_exits import entries, ex_fixed, SIX, W

plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
TP = 3.0


def build():
    df = get_frames()["H1"]
    idx, rows = entries(df)
    out = []
    for r in rows:
        Y, ei, ep, stop = r["Y"], r["ei"], r["ep"], r["stop"]
        xi, pr, why = ex_fixed(Y, ei, ep, stop, TP, 32)
        s = r["side"]
        Rp = ep - stop
        e_px = ep if s == 1 else -ep
        out.append(dict(cond=r["cond"], side=s, entry_time=idx[ei], exit_time=idx[xi], pnl_R=pr, exit=why,
                        entry_px=e_px, stop_px=e_px - s * Rp, target_px=e_px + s * TP * Rp, R_usd=Rp,
                        R_atr=Rp / Y["atr"][ei - 1], spr=Y["spr"][ei]))
    T = pd.DataFrame(out)
    T = pd.concat([nonoverlap(T[T.cond == k]) for k in SIX]).sort_values(["entry_time", "cond"]).reset_index(drop=True)
    T["pnl_usd"] = T.pnl_R * T.R_usd
    T["exit_px"] = T.entry_px + T.side * (T.pnl_usd + T.spr)
    T["combo_R"] = T.pnl_R * W
    T["hold_h"] = (T.exit_time - T.entry_time).dt.total_seconds() / 3600 + 1
    T["n_versions"] = T.groupby(["entry_time", "side"]).cond.transform("count")
    return df, T


def main():
    df, T = build()
    ec, em, nopen, worst = P.mtm_curve(df, T)
    o = []
    p = o.append
    r = T.combo_R
    days = pd.date_range(T.entry_time.min().normalize(), T.exit_time.max().normalize(), freq="B")
    daily = r.groupby(T.exit_time.dt.normalize()).sum().reindex(days, fill_value=0)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    S = T.groupby(["entry_time", "side"]).size()
    p("定案 H1 六合一（3R 停利），2023/4 ~ 2026/10，每版本 1/6 R")
    p(f"  交易 {len(T)} 筆（多 {(T.side == 1).sum()} / 空 {(T.side == -1).sum()}），訊號 {len(S)} 個（每週 {len(S) / (len(days) / 5):.2f}）")
    p(f"  勝率 {(T.pnl_R > 0).mean():.1%}；平均賺 {T.pnl_R[T.pnl_R > 0].mean():+.2f}R / 平均賠 {T.pnl_R[T.pnl_R <= 0].mean():+.2f}R；每筆 {T.pnl_R.mean():+.3f}R；PF {gp / gl:.2f}")
    p(f"  研究期每筆 {T[T.entry_time < SPLIT].pnl_R.mean():+.3f}R、驗證期 {T[T.entry_time >= SPLIT].pnl_R.mean():+.3f}R")
    p(f"  組合 {r.sum():+.2f}R；最大回撤 平倉 {(ec.cummax() - ec).max():.2f}R / 含浮動 {(em.cummax() - em).max():.2f}R（{(em.cummax() - em).idxmax():%Y-%m-%d}）")
    p(f"  日 Sharpe {daily.mean() / daily.std() * np.sqrt(252):.2f}；同時持倉最多 {nopen.max()} 筆；同時最大浮虧 {worst.min():+.2f}R")
    p("  逐年（組合 R）：" + "  ".join(f"{y}: {v:+.1f}" for y, v in r.groupby(T.entry_time.dt.year).sum().items()))
    M = r.groupby(T.exit_time.dt.to_period("M")).sum()
    M = M.reindex(pd.period_range(M.index.min(), M.index.max(), freq="M"), fill_value=0)
    p(f"  正報酬月 {(M > 0).sum()}/{len(M)}；最差月 {M.min():+.2f}R（{M.idxmin()}）；最好月 {M.max():+.2f}R（{M.idxmax()}）")
    for e, n in (("tp", "停利"), ("stop", "停損"), ("time", "時間")):
        g = T[T.exit == e]
        p(f"  出場 {n}: {len(g) / len(T):.0%}，平均 {g.pnl_R.mean():+.2f}R")
    p(f"  平均持有 {T.hold_h.mean():.1f} 小時；停損距離中位數 ${T.R_usd.median():.1f}（{T.R_atr.median():.2f} ATR）")
    for s, n in ((1, "多"), (-1, "空")):
        g = T[T.side == s]
        p(f"  {n}: {len(g)} 筆 每筆 {g.pnl_R.mean():+.3f}R 組合 {g.combo_R.sum():+.2f}R")
    p("\n美元（已扣進場點差、未扣隔夜利息）")
    X = T.copy(); X["combo_R"] = X.pnl_usd; X["R_usd"] = W
    c2, m2, _, w2 = P.mtm_curve(df, X)
    p(f"  每版本固定 0.01 手：總 ${T.pnl_usd.sum():,.0f}；最大回撤 含浮動 ${(m2.cummax() - m2).max():,.0f}；同時最大浮虧 ${-w2.min():,.0f}")
    p("    逐年 " + "  ".join(f"{y}: ${v:+,.0f}" for y, v in T.pnl_usd.groupby(T.entry_time.dt.year).sum().items()))
    for acct in (10000, 100000):
        u = acct * 0.01
        p(f"  帳戶 ${acct:,}、每訊號 1% 風險、不複利：總 ${r.sum() * u:+,.0f}（{r.sum() / 100:+.1%}），含浮動最大回撤約 ${(em.cummax() - em).max() * u:,.0f}；"
          f"手數中位數 {(u * W / (T.R_usd * 100)).median():.3f}")
    txt = "\n".join(o)
    print(txt)
    open("results_part32_final.txt", "w").write(txt + "\n")
    D = T[["cond", "side", "entry_time", "entry_px", "stop_px", "target_px", "R_usd", "R_atr", "exit_time", "exit_px", "exit",
           "hold_h", "pnl_usd", "pnl_R", "combo_R", "n_versions"]].copy()
    D["cond"] = D.cond.map(P.NAME); D["side"] = D.side.map({1: "多", -1: "空"}); D["exit"] = D.exit.map(P.EXIT)
    D.columns = ["版本", "方向", "進場時間", "進場價", "停損價", "停利價", "停損距離$", "停損距離ATR", "出場時間", "出場價", "出場原因",
                 "持有小時", "損益$每盎司", "損益R", "組合R", "同訊號版本數"]
    D.round({c: 3 for c in D.select_dtypes("number").columns}).to_csv("part32_final_trades.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(em.index, em.values, color="#9aa7b8", lw=0.8, label="含浮動")
    ax[0].plot(ec.index, ec.values, color="#1f5aa6", lw=1.6, label="已實現")
    ax[0].axvline(SPLIT, color="#888", ls="--", lw=1)
    ax[0].set_title("定案 H1 六合一（3R 停利，每版本 1/6 R）"); ax[0].set_ylabel("組合 R"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)
    d = em - em.cummax()
    ax[1].fill_between(d.index, d.values, 0, color="#c0504d", alpha=0.5); ax[1].set_ylabel("回撤 R"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("part32_final_equity.png", dpi=120)


if __name__ == "__main__":
    main()
