"""第二十八部分：H1 六合一（推進 / OBV / 速度衰減 / RSI / CCI / Stoch 背離順勢單同時開）逐筆明細與統計。

規則（每個版本各自獨立、各自一次只持有一筆）：
  事件  ：H1 突破前一個確認擺動高點（低點）且該版本背離成立（第二十二部分 detect）
  方向  ：順勢（跟著突破方向）
  過濾  ：日線狀態（SMA50 + 20 天斜率，前一天）與 H1 EMA200（前一根收盤）都和方向相同
  進場  ：訊號棒之後 4 根內突破訊號棒高點（做空 = 跌破低點）
  停損  ：第二推起點（回檔低點）外 0.1 ATR，且離進場至少 0.5 ATR
  出場  ：2R 停利、停損，或持有 32 根（32 小時）以收盤平倉；扣進場那根點差
  部位  ：每個版本每筆風險 1/6 R → 同一訊號 6 個版本都成立時合計 1R
輸出：
  part28_h1_combo_trades.csv   逐筆（每個版本一列）
  part28_h1_combo_signals.csv  依訊號合併（同進場時間、同方向）
  part28_h1_combo_monthly.csv  逐月
  results_part28_h1_combo.txt  統計
  part28_h1_combo_equity.png / part28_h1_combo_monthly.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_divergence import get_frames, nonoverlap, SPLIT
from part25_regime_drift import cont_trades

plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False
SIX = ["push", "obv", "speed", "rsi", "cci", "stoch"]
NAME = {"push": "推進", "obv": "OBV", "speed": "速度衰減", "rsi": "RSI", "cci": "CCI", "stoch": "Stoch"}
W = 1 / 6
REG = {1: "多頭", 0: "盤整", -1: "空頭"}
EXIT = {"tp": "停利", "stop": "停損", "time": "時間"}


def build():
    df = get_frames()["H1"]
    A, _ = cont_trades(df)
    A = A[(A.side == A.reg) & (A.side == A.ema) & A.cond.isin(SIX)]
    T = pd.concat([nonoverlap(A[A.cond == k]) for k in SIX]).sort_values(["entry_time", "cond"]).reset_index(drop=True)
    s = T.side
    R_px = T.ep_y - T.stop_y
    T["entry_px"] = np.where(s == 1, T.ep_y, -T.ep_y)
    T["stop_px"] = np.where(s == 1, T.stop_y, -T.stop_y)
    T["target_px"] = T.entry_px + s * 2 * R_px
    T["R_usd"] = R_px
    T["pnl_usd"] = T.pnl_R * R_px                      # 每盎司（= 0.01 手）美元，已扣點差
    T["exit_px"] = T.entry_px + s * (T.pnl_usd + T.spr)
    T["hold_h"] = (T.exit_time - T.entry_time).dt.total_seconds() / 3600 + 1
    T["combo_R"] = T.pnl_R * W
    cnt = T.groupby(["entry_time", "side"]).cond.transform("count")
    T["n_versions"] = cnt
    # 帳戶 1 萬美元、每個版本風險 1/6 %（同一訊號六個版本全中 = 1%）的手數
    T["lots_10k_1pct"] = (10000 * 0.01 * W / (R_px * 100)).round(3)
    return df, T


def mtm_curve(df, T):
    """逐小時的組合淨值（已實現 + 未實現，未實現用 K 棒最差價），單位：組合 R。"""
    idx = df.index
    H, L, C = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    pos = pd.Series(np.arange(len(idx)), index=idx)
    real = np.zeros(len(idx))
    worst = np.zeros(len(idx))
    close_u = np.zeros(len(idx))
    nopen = np.zeros(len(idx), int)
    for r in T.itertuples():
        i0, i1 = pos[r.entry_time], pos[r.exit_time]
        real[i1] += r.combo_R
        if i1 > i0:
            k = np.arange(i0, i1)
            px = L[k] if r.side == 1 else H[k]
            worst[k] += r.side * (px - r.entry_px) / r.R_usd * W
            close_u[k] += r.side * (C[k] - r.entry_px) / r.R_usd * W
            nopen[k] += 1
    cum = np.cumsum(real)
    eq_closed = pd.Series(cum, index=idx)
    eq_mtm = pd.Series(cum + worst, index=idx)
    return eq_closed, eq_mtm, pd.Series(nopen, index=idx), pd.Series(worst, index=idx)


def summary(x, col="combo_R"):
    r = x[col]
    if len(r) == 0:
        return "—"
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return (f"{len(r):4d} 筆  勝率 {(r > 0).mean():5.1%}  每筆 {x.pnl_R.mean():+.3f}R  "
            f"組合合計 {r.sum():+6.2f}R  PF {gp / gl if gl else np.inf:4.2f}")


def main():
    df, T = build()
    eq_c, eq_m, nopen, worst = mtm_curve(df, T)
    out = []
    p = out.append
    p("H1 六合一（推進 / OBV / 速度衰減 / RSI / CCI / Stoch 背離順勢單，D+E 盤面過濾）")
    p(f"期間 {T.entry_time.min():%Y-%m-%d} ~ {T.exit_time.max():%Y-%m-%d}；每個版本每筆 1/6 R；研究期 2023~24、驗證期 2025~26/10")
    p("R = 該筆停損距離；「組合 R」= 該筆 R × 1/6（同一訊號 6 個版本全中合計 1R）")

    p("\n## 1. 總覽")
    r = T.combo_R
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    dd_c = (eq_c.cummax() - eq_c).max()
    dd_m = (eq_m.cummax() - eq_m).max()
    days = pd.date_range(T.entry_time.min().normalize(), T.exit_time.max().normalize(), freq="B")
    daily = T.groupby(T.exit_time.dt.normalize()).combo_R.sum().reindex(days, fill_value=0)
    S = T.groupby(["entry_time", "side"]).agg(n=("cond", "count"), R=("combo_R", "sum"), pnl_R=("pnl_R", "mean")).reset_index()
    p(f"  交易筆數（版本層級）      {len(T)}（多 {(T.side == 1).sum()} / 空 {(T.side == -1).sum()}）")
    p(f"  不同訊號數（同進場時間+方向） {len(S)}；平均每週 {len(S) / (len(days) / 5):.2f} 個訊號")
    p(f"  勝率                      {(T.pnl_R > 0).mean():.1%}（訊號層級 {(S.R > 0).mean():.1%}）")
    p(f"  平均賺 / 平均賠（R）      {T.pnl_R[T.pnl_R > 0].mean():+.2f} / {T.pnl_R[T.pnl_R <= 0].mean():+.2f}")
    p(f"  每筆期望（R）             {T.pnl_R.mean():+.3f}；中位數 {T.pnl_R.median():+.3f}")
    p(f"  PF                        {gp / gl:.2f}")
    p(f"  組合總損益                {r.sum():+.2f} R")
    p(f"  最大回撤（平倉）          {dd_c:.2f} R")
    p(f"  最大回撤（含浮動，逐小時最差價） {dd_m:.2f} R（{(eq_m.cummax() - eq_m).idxmax():%Y-%m-%d %H:%M}）")
    p(f"  總損益 / 含浮動回撤       {r.sum() / dd_m:.1f}")
    p(f"  日 Sharpe（全部平日）     {daily.mean() / daily.std() * np.sqrt(252):.2f}")
    p(f"  正報酬日（有出場的日子）  {(daily[daily != 0] > 0).mean():.1%}")
    p(f"  最好 / 最差單日           {daily.max():+.2f} / {daily.min():+.2f} R")
    p(f"  同時持倉 最多 / 平均       {nopen.max()} 筆 / {nopen[nopen > 0].mean():.1f} 筆（有持倉時）；同時最大浮虧 {worst.min():+.2f} R")
    lose = (T.sort_values("exit_time").pnl_R <= 0).astype(int).to_numpy()
    runs = np.diff(np.flatnonzero(np.r_[1, np.diff(lose) != 0, 1]))
    vals = lose[np.flatnonzero(np.r_[1, np.diff(lose) != 0])]
    p(f"  最長連虧 / 連勝（筆）     {runs[vals == 1].max()} / {runs[vals == 0].max()}")
    p(f"  持有時間 平均 / 中位數     {T.hold_h.mean():.1f} / {T.hold_h.median():.0f} 小時")
    p(f"  停損距離 中位數           ${T.R_usd.median():.1f}（{T.R_atr.median():.2f} ATR），範圍 ${T.R_usd.min():.1f} ~ ${T.R_usd.max():.1f}")

    p("\n## 2. 研究期 / 驗證期 / 逐年")
    p(f"  研究期 2023~24 : {summary(T[T.entry_time < SPLIT])}")
    p(f"  驗證期 2025~26 : {summary(T[T.entry_time >= SPLIT])}")
    for y, g in T.groupby(T.entry_time.dt.year):
        e = eq_c[eq_c.index.year == y]
        e = e - (eq_c[eq_c.index < f"{y}-01-01"].iloc[-1] if (eq_c.index < f"{y}-01-01").any() else 0)
        p(f"  {y}           : {summary(g)}  年內最大回撤 {(e.cummax() - e).max():.2f}R")

    p("\n## 3. 依版本")
    for k in SIX:
        p(f"  {NAME[k]:6s}: {summary(T[T.cond == k])}")

    p("\n## 4. 多 / 空，日線狀態")
    for s, sn in ((1, "多"), (-1, "空")):
        p(f"  {sn}: {summary(T[T.side == s])}")

    p("\n## 5. 出場方式")
    for e in ("tp", "stop", "time"):
        g = T[T.exit == e]
        p(f"  {EXIT[e]}: {len(g):4d} 筆（{len(g) / len(T):.0%}） 平均 {g.pnl_R.mean():+.2f}R  平均持有 {g.hold_h.mean():.1f} 小時")

    p("\n## 6. 同一訊號有幾個版本同時成立（訊號層級，合計組合 R）")
    for n, g in S.groupby("n"):
        p(f"  {n} 個版本: {len(g):4d} 個訊號  勝率 {(g.R > 0).mean():.1%}  平均每筆 {g.pnl_R.mean():+.3f}R  合計 {g.R.sum():+6.2f} 組合R")

    p("\n## 7. 持有時間")
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 24), (24, 40)):
        g = T[(T.hold_h > lo) & (T.hold_h <= hi)]
        p(f"  {lo:2d}~{hi:2d} 小時: {len(g):4d} 筆  勝率 {(g.pnl_R > 0).mean():.1%}  平均 {g.pnl_R.mean():+.2f}R")

    p("\n## 8. 進場時段（broker 時間）")
    T["hr"] = T.entry_time.dt.hour
    for lo, hi in ((0, 7), (7, 12), (12, 17), (17, 24)):
        g = T[(T.hr >= lo) & (T.hr < hi)]
        p(f"  {lo:02d}~{hi:02d} 時: {summary(g)}")

    p("\n## 9. 逐月（組合 R）")
    T["ym"] = T.exit_time.dt.to_period("M")
    M = T.groupby("ym").agg(筆數=("combo_R", "size"), 勝率=("pnl_R", lambda x: (x > 0).mean()),
                            組合R=("combo_R", "sum"), 多=("side", lambda x: (x == 1).sum()), 空=("side", lambda x: (x == -1).sum()))
    allm = pd.period_range(T.ym.min(), T.ym.max(), freq="M")
    M = M.reindex(allm).fillna({"筆數": 0, "組合R": 0, "多": 0, "空": 0})
    M["累計"] = M.組合R.cumsum()
    p(f"  正報酬月 {(M.組合R > 0).sum()} / {len(M)}；最好月 {M.組合R.max():+.2f}R、最差月 {M.組合R.min():+.2f}R")
    p(M.round(2).to_string())
    M.round(3).to_csv("part28_h1_combo_monthly.csv", index_label="月份")

    p("\n## 10. 部位換算（1 萬美元帳戶、每個版本風險 1/6 %，同一訊號六版全中 = 1%）")
    p(f"  手數 = 10000 × 1% × 1/6 ÷（停損距離 × 100）；中位數 {T.lots_10k_1pct.median():.3f} 手（多數券商最小 0.01 手，需要較大帳戶或改用固定 0.01 手）")
    p(f"  以這個風險換算：總報酬約 {r.sum():+.1f}%，含浮動最大回撤約 {dd_m:.1f}%（不複利）")

    txt = "\n".join(out)
    open("results_part28_h1_combo.txt", "w").write(txt + "\n")
    print(txt)

    cols = ["cond", "side", "sig_time", "p1_time", "entry_time", "entry_px", "stop_px", "target_px", "R_usd", "R_atr",
            "exit_time", "exit_px", "exit", "hold_h", "pnl_usd", "pnl_R", "combo_R", "n_versions", "reg", "lots_10k_1pct"]
    D = T[cols].copy()
    D["cond"] = D.cond.map(NAME)
    D["side"] = D.side.map({1: "多", -1: "空"})
    D["exit"] = D.exit.map(EXIT)
    D["reg"] = D.reg.map(REG)
    D = D.rename(columns={"cond": "版本", "side": "方向", "sig_time": "訊號K棒", "p1_time": "前一擺動點", "entry_time": "進場時間",
                          "entry_px": "進場價", "stop_px": "停損價", "target_px": "停利價", "R_usd": "停損距離$", "R_atr": "停損距離ATR",
                          "exit_time": "出場時間", "exit_px": "出場價", "exit": "出場原因", "hold_h": "持有小時", "pnl_usd": "損益$每盎司",
                          "pnl_R": "損益R", "combo_R": "組合R", "n_versions": "同訊號版本數", "reg": "日線狀態", "lots_10k_1pct": "手數(1萬,1%/6)"})
    D.round({c: 3 for c in D.select_dtypes("number").columns}).to_csv("part28_h1_combo_trades.csv", index=False, encoding="utf-8-sig")
    S["side"] = S.side.map({1: "多", -1: "空"})
    S.rename(columns={"entry_time": "進場時間", "side": "方向", "n": "版本數", "R": "組合R", "pnl_R": "平均損益R"}).round({"組合R": 3, "平均損益R": 3}).to_csv(
        "part28_h1_combo_signals.csv", index=False, encoding="utf-8-sig")
    M.round(3).to_csv("part28_h1_combo_monthly.csv", index_label="月份", encoding="utf-8-sig")

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(eq_m.index, eq_m.values, color="#9aa7b8", lw=0.8, label="含浮動（逐小時最差價）")
    ax[0].plot(eq_c.index, eq_c.values, color="#1f5aa6", lw=1.6, label="已實現")
    ax[0].axvline(SPLIT, color="#888", ls="--", lw=1)
    ax[0].text(SPLIT, ax[0].get_ylim()[1] * 0.95, "  驗證期", color="#555")
    ax[0].set_ylabel("組合 R")
    ax[0].set_title("H1 六合一 背離順勢單（每版本 1/6 R）")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ddm = eq_m - eq_m.cummax()
    ax[1].fill_between(ddm.index, ddm.values, 0, color="#c0504d", alpha=0.5)
    ax[1].set_ylabel("回撤 R")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("part28_h1_combo_equity.png", dpi=120)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = M.index.to_timestamp()
    ax.bar(x, M.組合R, width=20, color=np.where(M.組合R >= 0, "#1f5aa6", "#c0504d"))
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("組合 R")
    ax.set_title("H1 六合一 逐月損益")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("part28_h1_combo_monthly.png", dpi=120)


if __name__ == "__main__":
    main()
