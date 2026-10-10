"""第五十部分：很陡順勢的樣本外驗證（2023 ~ 2025），參數完全不改。

資料：data/XAUUSD_M15_2023_2026.csv（M15，2023/1/3 ~ 2026/10/9；和 M1 重採樣的 2026 重疊部分收盤完全一致）。
第 49 部分全部只用 2026/1 ~ 10 → 2023、2024、2025 是沒看過的資料。
事先固定要驗證的東西
  主設定（49j）：L48、EWMA z(cosθ) W=1000 ≤ −2、順勢、1.5 ATR 停損、48 根出場
  鄰近 32 格（49j）：L 48/56/64/96 × SMA/EWMA × W 500/1000 × 門檻 −1.5/−2
  對照：μ|a|（49e：SMA W500 L48 ≤ −2）、滑（49f：SMA W500 L48 z(ln μ) ≤ −1）
  基準：同 L 每根都順勢、同樣出場
z 在整段資料上連續計算，交易依進場年份分組。
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import features, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma, z_ewm

PATH = "data/XAUUSD_M15_2023_2026.csv"
YEARS = [2023, 2024, 2025, 2026]


def summ(T):
    if len(T) < 5:
        return dict(n=len(T))
    m, t, _ = tst(T.R.to_numpy())
    p = T.pnl
    eq = p.cumsum()
    dd = (eq - np.maximum.accumulate(np.maximum(eq, 0))).min()
    return dict(n=len(T), R=m, t=t, win=(p > 0).mean(), pf=p[p > 0].sum() / -p[p < 0].sum(), usd=p.mean(),
                tot=p.sum(), dd=dd, top10=p.nlargest(10).sum() / p.sum() if p.sum() > 0 else np.nan)


def fmt(s, base=None):
    if "R" not in s:
        return f"n={s['n']:3d}"
    ex = f" 超{s['R'] - base:+.2f}" if base is not None else ""
    return (f"n={s['n']:3d} 勝{s['win']:.0%} {s['R']:+.3f}R(t={s['t']:+.1f}){ex} PF{s['pf']:.2f} "
            f"每筆{s['usd']:+6.2f} 總{s['tot']:+6.0f} DD{s['dd']:5.0f}")


def by_year(T):
    return {y: T[T.entry_time.dt.year == y] for y in YEARS} | {"23-25": T[T.entry_time.dt.year <= 2025]}


def main():
    df = load(PATH)
    out = ["第五十部分：很陡順勢 樣本外驗證（參數不改）", f"資料 {df.index.min()} ~ {df.index.max()}，{len(df)} 根 M15", ""]
    Fs, bases, basesT = {}, {}, {}
    for L in (48, 56, 64, 96):
        F = features(df, L)
        Fs[L] = F
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        B = backtest(df, F, valid, L)
        basesT[L] = B
        bases[L] = {k: (v.R.mean() if len(v) else np.nan) for k, v in by_year(B).items()}

    def run(L, x, zf, w, thr):
        F = Fs[L]
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        return backtest(df, F, valid & (zf(x, w) <= thr), L)

    F = Fs[48]
    main_T = run(48, F.cos, z_ewm, 1000, -2.0)
    main_T.to_csv("part50_trades_main.csv", index=False, float_format="%.4f")
    rules = {
        "主設定 很陡 EWMA W1000 ≤−2": main_T,
        "很陡 SMA W500 ≤−2（49 最好那格）": run(48, F.cos, z_sma, 500, -2.0),
        "μ|a| SMA W500 ≤−2（49e t=3.0）": run(48, np.log(F.mu) + np.log(F.a.abs().clip(lower=1e-6)), z_sma, 500, -2.0),
        "滑 z(ln μ) SMA W500 ≤−1（49f）": run(48, np.log(F.mu), z_sma, 500, -1.0),
    }
    out.append("===== 各規則逐年（L48；超額 = 每筆 R − 同年基準）=====")
    out.append("基準 全部順勢 L48：")
    for k, v in by_year(basesT[48]).items():
        out.append(f"  {k}: {fmt(summ(v))}")
    for name, T in rules.items():
        out.append(f"{name}：")
        for k, v in by_year(T).items():
            out.append(f"  {k}: {fmt(summ(v), bases[48][k])}")
    out.append("")

    # 鄰近 32 格
    rows = []
    for L in (48, 56, 64, 96):
        x = Fs[L].cos
        for zn, zf in (("EWMA", z_ewm), ("SMA", z_sma)):
            for w in (500, 1000):
                for thr in (-1.5, -2.0):
                    T = run(L, x, zf, w, thr)
                    for k, v in by_year(T).items():
                        s = summ(v)
                        if "R" in s:
                            rows.append(dict(L=L, z=zn, W=w, thr=thr, per=k, n=s["n"], R=s["R"], ex=s["R"] - bases[L][k],
                                             pf=s["pf"], tot=s["tot"]))
    G = pd.DataFrame(rows)
    G.to_csv("part50_grid.csv", index=False, float_format="%.4f")
    out.append("===== 鄰近 32 格（很陡）逐年彙總 =====")
    agg = G.groupby("per").agg(格數=("R", "size"), R正比例=("R", lambda v: (v > 0).mean()), 平均R=("R", "mean"),
                               超額正比例=("ex", lambda v: (v > 0).mean()), 平均超額=("ex", "mean"),
                               最差超額=("ex", "min"), PF大於1比例=("pf", lambda v: (v > 1).mean()), 平均筆數=("n", "mean"))
    out.append(agg.reindex(["2023", "2024", "2025", "23-25", "2026"] if False else [2023, 2024, 2025, "23-25", 2026]).round(3).to_string())
    out.append("")
    out.append("各 L 的 2023-25 平均超額 / 2026 平均超額：")
    for L in (48, 56, 64, 96):
        a = G[(G.L == L) & (G.per == "23-25")].ex
        b = G[(G.L == L) & (G.per == 2026)].ex
        out.append(f"  L={L}: 23-25 {a.mean():+.3f}（正 {(a > 0).mean():.0%}） | 2026 {b.mean():+.3f}（正 {(b > 0).mean():.0%}）")
    out.append("")
    out.append("基準 全部順勢 各 L 逐年每筆 R：")
    for L in (48, 56, 64, 96):
        out.append(f"  L={L}: " + "  ".join(f"{k} {v:+.3f}" for k, v in bases[L].items()))
    txt = "\n".join(out)
    print(txt)
    with open("results_part50.txt", "w") as fh:
        fh.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(main_T.exit_time, main_T.R.cumsum(), color="#2a6fdb", lw=1.6, label=f"很陡順勢 主設定（{len(main_T)} 筆）")
    B = basesT[48]
    ax[0].plot(B.exit_time, B.R.cumsum() * len(main_T) / len(B), color="#999", lw=1,
               label="基準 全部順勢（累積 R × 筆數比例縮放）")
    ax[0].axvline(pd.Timestamp("2026-01-01"), color="#c33", ls="--", lw=0.8)
    ax[0].text(pd.Timestamp("2026-01-10"), ax[0].get_ylim()[0], " 第 49 部分研究期間 →", color="#c33", va="bottom")
    ax[0].set_title("M15 L48 很陡順勢：累積 R（2023 ~ 2025 為樣本外）")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    piv = G[G.per.isin(YEARS)].groupby("per").ex.apply(list)
    ax[1].boxplot([piv[y] for y in YEARS], tick_labels=[str(y) for y in YEARS])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("鄰近 32 格的超額 R（每筆 R - 同年全部順勢）")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("part50_oos.png", dpi=110)


if __name__ == "__main__":
    main()
