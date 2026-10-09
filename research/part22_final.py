"""第二十二部分 I：通用回復力指數（RFI-U）與雙向交易的最終評估。

量測挑選（只用 IS）：在 4 個「不吸收趨勢」的均值（EMA20、EMA50、KAMA、滾動 VWAP20）中，
IS1 與 IS2 的 Δβ 同向且兩段都 ≥ 0.02 的次數 ≥ 3 次、方向一致 → 入選（見 part22_beta.csv 的 IS 欄）。
入選 8 個：ER20(−)、ER5/ER20(+)、離開速度(+)、離開根數(−)、日波幅已用(+)、當天 ER 順向(−)、VWAP 同側距(−)、順高週期(−)。
RFI-U = 平均( 方向 × (IS 分位 − 0.5) )。高 = 回復力強（做回歸），低 = 回復力弱（做動能）。

交易評估以「事件」為單位（|D| 第一次 ≥ d0，回到 d0/2 內才重新武裝），每個事件同時計算兩種結果：
  fade = 反向、follow = 順向；下一根開盤進場；持有 H 根或 2 ATR 停損；各自扣進場點差。
報告：全部事件（基準）、RFI 高三分位做回歸、RFI 低三分位做動能、以及相對基準的增益；
OOS 增益以「天」為單位 bootstrap 給 90% 區間。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_lib import load_tf, means, atr, events, ema, CUT
from part22_beta import frame, state_beta, beta
from part22_restore import NAMES

out = open("results_part22.txt", "a")
def P(s=""): print(s, flush=True); out.write(s + "\n"); out.flush()
CUT1 = pd.Timestamp("2026-04-01")
CLASSIC = ["ema20", "ema50", "kama", "rvwap20"]


def universal_selection(S, thr=0.02, need=3):
    """S = part22_beta.csv；只用 db_IS1 / db_IS2。"""
    S = S[S["mean"].isin(CLASSIC) & (S.feat != "dist")].copy()
    S["ok"] = (np.sign(S.db_IS1) == np.sign(S.db_IS2)) & (np.minimum(S.db_IS1.abs(), S.db_IS2.abs()) >= thr)
    keep = []
    for f, g in S[S.ok].groupby("feat"):
        sg = np.sign(g.db_IS1)
        if len(g) >= need and abs(sg.sum()) == len(g):
            keep.append((f, int(sg.iloc[0])))
    return sorted(keep)


def rfi_u(X, keep):
    IS = X.per != "OOS"
    parts = []
    for f, s in keep:
        ref = np.sort(X.loc[IS, f].dropna().to_numpy())
        parts.append(s * (pd.Series(np.searchsorted(ref, X[f].to_numpy()) / len(ref), X.index).where(X[f].notna()) - 0.5))
    return pd.concat(parts, axis=1).mean(axis=1)


def outcomes(d, A, ev, sgn, H, stop=2.0):
    """sgn = 交易方向；回傳 ATR 單位的淨損益。"""
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = d.spread.to_numpy() / 100; seg = d.seg.to_numpy(); a = A.to_numpy(); n = len(c)
    res = np.full(len(ev), np.nan)
    for k, t in enumerate(ev):
        e = t + 1
        if e + H >= n:
            continue
        s = sgn[k]; ep = o[e]; st = ep - s * stop * a[t]; xp = None
        for j in range(e, e + H):
            if seg[j] != seg[t]:
                xp = c[j - 1]; break
            if (l[j] <= st) if s > 0 else (h[j] >= st):
                xp = (min(st, o[j]) if s > 0 else max(st, o[j])) if j > e else st; break
        if xp is None:
            xp = c[e + H - 1]
        res[k] = (s * (xp - ep) - spr[e]) / a[t]
    return res


def boot_diff(x_sel, day_sel, x_all, day_all, n=2000, seed=0):
    """以天為單位重抽：mean(sel) − mean(all)。"""
    rng = np.random.default_rng(seed)
    days = np.unique(day_all)
    gs = pd.Series(x_sel).groupby(day_sel).agg(["sum", "count"]).reindex(days).fillna(0)
    ga = pd.Series(x_all).groupby(day_all).agg(["sum", "count"]).reindex(days).fillna(0)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(days), len(days))
        s, a = gs.iloc[idx].sum(), ga.iloc[idx].sum()
        if s["count"] > 0 and a["count"] > 0:
            out.append(s["sum"] / s["count"] - a["sum"] / a["count"])
    return np.quantile(out, [0.05, 0.95])


def event_table(d, A, X, d0, Hs=(6, 12, 24)):
    D = X.D.reindex(d.index)
    ev = events(D, d.seg, d0=d0, rearm=d0 / 2)
    ev = ev[ev > 600]
    sd = np.sign(D.to_numpy()[ev])
    E = pd.DataFrame({"t": d.index[ev], "D": D.to_numpy()[ev], "rfi": X.rfi.reindex(d.index).to_numpy()[ev]})
    for H in Hs:
        E[f"fade{H}"] = outcomes(d, A, ev, -sd, H)
        E[f"follow{H}"] = outcomes(d, A, ev, sd, H)
    E["per"] = np.where(E.t < CUT1, "IS1", np.where(E.t < CUT, "IS2", "OOS"))
    E["day"] = E.t.dt.normalize()
    E["sess"] = np.select([E.t.dt.hour < 10, E.t.dt.hour < 15, E.t.dt.hour < 20], ["ASIA", "LDN", "NY"], "LATE")
    return E.dropna(subset=["rfi"])


if __name__ == "__main__":
    P("\n" + "=" * 120)
    S = pd.read_csv("part22_beta.csv")
    keep = universal_selection(S)
    P("第二十二部分 I：通用回復力指數 RFI-U（只用 IS 挑選）")
    P("  入選：" + "、".join(f"{NAMES[f]}({'+' if s > 0 else '−'})" for f, s in keep))
    d = load_tf("M5"); A = atr(d, 14); M = means(d)

    P("\n1. RFI-U 五分位的回復係數 β（M5 全部 K 棒，前瞻 12 根）")
    Xs = {}
    for mn in ["ema20", "ema50", "kama", "rvwap20", "sma20", "ema100", "linreg20"]:
        X = frame(d, A, M, mn); X["rfi"] = rfi_u(X, keep); Xs[mn] = X
        t = state_beta(X, "rfi")
        P(f"  {mn:9s} " + " | ".join(f"{p} " + " ".join(f"{v:+.2f}" for v in t[p]) + f" Δ{t[p][-1] - t[p][0]:+.2f}" for p in ("IS1", "IS2", "OOS"))
          + ("  （未參與挑選）" if mn not in CLASSIC else ""))

    P("\n2. 跨週期（同物理時間的 EMA，前瞻 60 分鐘；量測與方向沿用 M5 的挑選）")
    for tf, n, H in (("M1", 100, 60), ("M1", 250, 60), ("M15", 7, 4), ("M15", 17, 4)):
        dd = load_tf(tf); AA = atr(dd, 14)
        X = frame(dd, AA, pd.DataFrame({"x": ema(dd.close, n)}), "x", H=H); X["rfi"] = rfi_u(X, keep)
        t = state_beta(X, "rfi")
        P(f"  {tf:3s} EMA{n:<4d} " + " | ".join(f"{p} " + " ".join(f"{v:+.2f}" for v in t[p]) + f" Δ{t[p][-1] - t[p][0]:+.2f}" for p in ("IS1", "IS2", "OOS")))

    P("\n3. 事件交易：高 RFI 三分位做回歸、低 RFI 三分位做動能（ATR/筆，扣點差；增益 = 篩選 − 全部事件同方向）")
    rows = []
    for mn in CLASSIC + ["sma20", "ema100"]:
        X = Xs[mn]
        lo, hi = np.nanquantile(X.loc[X.per != "OOS", "rfi"], [1 / 3, 2 / 3])
        for d0 in (1.5, 2.0, 2.5):
            E = event_table(d, A, X, d0)
            for H in (6, 12, 24):
                for leg, col, m in (("回歸", f"fade{H}", E.rfi >= hi), ("動能", f"follow{H}", E.rfi <= lo)):
                    r = dict(mean=mn, d0=d0, H=H, leg=leg)
                    for p in ("IS1", "IS2", "OOS"):
                        sel, al = E[(E.per == p) & m], E[E.per == p]
                        r[f"n_{p}"], r[f"sel_{p}"], r[f"all_{p}"] = len(sel), sel[col].mean(), al[col].mean()
                        r[f"gain_{p}"] = r[f"sel_{p}"] - r[f"all_{p}"]
                    o_sel, o_all = E[(E.per == "OOS") & m], E[E.per == "OOS"]
                    r["ci_lo"], r["ci_hi"] = boot_diff(o_sel[col].to_numpy(), o_sel.day.to_numpy(), o_all[col].to_numpy(), o_all.day.to_numpy(), n=500)
                    rows.append(r)
    G = pd.DataFrame(rows)
    G.to_csv("part22_final_grid.csv", index=False)
    for (mn, leg), g in G.groupby(["mean", "leg"], sort=False):
        P(f"\n  --- {mn} {leg} ---   d0   H  | IS1 n 篩選/全部 | IS2 n 篩選/全部 | OOS n 篩選/全部 | OOS 增益 [90%]")
        for r in g.itertuples():
            P(f"  {'':18s} {r.d0:.1f} {r.H:3d} | {r.n_IS1:4d} {r.sel_IS1:+.3f}/{r.all_IS1:+.3f} | {r.n_IS2:4d} {r.sel_IS2:+.3f}/{r.all_IS2:+.3f} | "
              f"{r.n_OOS:4d} {r.sel_OOS:+.3f}/{r.all_OOS:+.3f} | {r.gain_OOS:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
    P("\n  總覽（9 組 d0×H 平均）：增益 IS1 / IS2 / OOS、OOS 增益 > 0 的比例、OOS 篩選後絕對報酬 > 0 的比例")
    for (mn, leg), g in G.groupby(["mean", "leg"], sort=False):
        P(f"  {mn:8s} {leg}  增益 {g.gain_IS1.mean():+.3f} / {g.gain_IS2.mean():+.3f} / {g.gain_OOS.mean():+.3f}   "
          f"OOS 增益>0 {np.mean(g.gain_OOS > 0):.0%}   OOS 絕對>0 {np.mean(g.sel_OOS > 0):.0%}   （參與挑選）" if mn in CLASSIC else
          f"  {mn:8s} {leg}  增益 {g.gain_IS1.mean():+.3f} / {g.gain_IS2.mean():+.3f} / {g.gain_OOS.mean():+.3f}   "
          f"OOS 增益>0 {np.mean(g.gain_OOS > 0):.0%}   OOS 絕對>0 {np.mean(g.sel_OOS > 0):.0%}   （未參與挑選）")

    P("\n3b. 雙向切換 vs 單邊策略（同一批事件；RFI 中間三分位不做）：每筆 ATR（t 值）")
    P("    以『兩段都要能用』為目標：單邊策略在 IS 與 OOS 的贏家不同（IS 回歸、OOS 動能），切換版是否兩段都正？")
    rows3 = []
    for mn in CLASSIC:
        X = Xs[mn]; lo, hi = np.nanquantile(X.loc[X.per != "OOS", "rfi"], [1 / 3, 2 / 3])
        for d0 in (1.5, 2.0, 2.5):
            E = event_table(d, A, X, d0)
            for H in (6, 12, 24):
                sw = np.where(E.rfi >= hi, E[f"fade{H}"], np.where(E.rfi <= lo, E[f"follow{H}"], np.nan))
                r = dict(mean=mn, d0=d0, H=H)
                for p, m in (("IS", E.per != "OOS"), ("OOS", E.per == "OOS")):
                    x = pd.Series(sw[m.to_numpy()]).dropna()
                    r[f"sw_{p}"], r[f"t_{p}"] = x.mean(), x.mean() / x.std() * np.sqrt(len(x))
                    r[f"fade_{p}"], r[f"follow_{p}"] = E[m][f"fade{H}"].mean(), E[m][f"follow{H}"].mean()
                rows3.append(r)
    G3 = pd.DataFrame(rows3); G3.to_csv("part22_switch_grid.csv", index=False)
    for mn, g in G3.groupby("mean", sort=False):
        P(f"  {mn:8s} 切換 IS {g.sw_IS.mean():+.3f}（正 {np.mean(g.sw_IS > 0):.0%}，t 中位 {g.t_IS.median():+.1f}） OOS {g.sw_OOS.mean():+.3f}（正 {np.mean(g.sw_OOS > 0):.0%}，t 中位 {g.t_OOS.median():+.1f}）"
          f" | 永遠回歸 IS {g.fade_IS.mean():+.3f} OOS {g.fade_OOS.mean():+.3f} | 永遠動能 IS {g.follow_IS.mean():+.3f} OOS {g.follow_OOS.mean():+.3f}")

    # 4. 代表設定：EMA20、d0 2.0、H 12 的細節（時段、月份）＋ 圖
    X = Xs["ema20"]; lo, hi = np.nanquantile(X.loc[X.per != "OOS", "rfi"], [1 / 3, 2 / 3])
    E = event_table(d, A, X, 2.0)
    E["leg"] = np.where(E.rfi >= hi, "MR", np.where(E.rfi <= lo, "MOM", "-"))
    E["pnl"] = np.where(E.leg == "MR", E.fade12, np.where(E.leg == "MOM", E.follow12, np.nan))
    E.to_csv("part22_events_ema20.csv", index=False)
    P("\n4. 代表設定 EMA20、d0 = 2.0、H = 12：分時段 / 分月（ATR/筆）")
    for leg, lab in (("MR", "回歸"), ("MOM", "動能")):
        sub = E[E.leg == leg]
        P(f"  {lab}：" + "  ".join(f"{s} {len(g)}筆 {g.pnl.mean():+.3f}" for s, g in sub.groupby("sess")))
        P(f"        " + "  ".join(f"{k.month}月 {g.pnl.mean():+.2f}({len(g)})" for k, g in sub.groupby(sub.t.dt.to_period("M"))))
    both = E[E.leg != "-"].sort_values("t")
    P(f"  合併（兩邊都做）：IS {both[both.per != 'OOS'].pnl.mean():+.3f}（{(both.per != 'OOS').sum()} 筆）  OOS {both[both.per == 'OOS'].pnl.mean():+.3f}（{(both.per == 'OOS').sum()} 筆）")
    allf = E.sort_values("t")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for leg, lab, col in (("MR", "high RFI -> fade", "tab:green"), ("MOM", "low RFI -> follow", "tab:red")):
        sub = E[E.leg == leg].sort_values("t")
        ax[0].plot(sub.t, sub.pnl.cumsum(), label=lab, color=col)
    ax[0].plot(allf.t, allf.fade12.cumsum(), label="all events fade", color="gray", ls="--")
    ax[0].plot(allf.t, allf.follow12.cumsum(), label="all events follow", color="black", ls=":")
    ax[0].plot(both.t, both.pnl.cumsum(), label="switch (both legs)", color="tab:blue", lw=2)
    ax[0].axvline(CUT, color="k", lw=.8); ax[0].legend(); ax[0].set_title("EMA20, |D|>=2 ATR, hold 12 x M5 (cum ATR)")
    for p, c in (("IS1", "tab:blue"), ("IS2", "tab:orange"), ("OOS", "tab:purple")):
        t = state_beta(X, "rfi")
        ax[1].plot(range(1, 6), t[p], marker="o", label=p, color=c)
    ax[1].axhline(0, color="k", lw=.6); ax[1].set_xlabel("RFI-U quintile (1 = weak restoring, 5 = strong)")
    ax[1].set_ylabel("beta (fraction of D recovered in 12 bars)"); ax[1].legend(); ax[1].set_title("State-dependent restoring coefficient, EMA20 M5")
    fig.tight_layout(); fig.savefig("part22_rfi.png", dpi=110)
