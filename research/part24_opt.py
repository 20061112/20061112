"""第二十四部分：回復力切換策略的優化（持倉限制、參數、因子調配）。

原則：所有選擇只用 IS（1~5 月，再分 IS1 = 1~3 月、IS2 = 4~5 月），OOS（6~10 月）只驗證；
同時列出整個參數格子在 OOS 的分佈（高原 vs 尖點）。
金額 = 美元/盎司（0.01 手），已扣進場點差。
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_lib import CUT
from part24_lib import load_all, event_frame, add_outcomes, apply_cap, summarize, fmt, CUT1

def P(s=""):
    print(s, flush=True)
    with open("results_part24.txt", "a") as f:
        f.write(s + "\n")

MR_SPECS = {f"mr_H{H}_s{st}": ("mr", H, st, False) for H in (6, 12, 24) for st in (1.5, 2.0, 3.0)}
MR_SPECS.update({f"mr_T_H{H}_s{st}": ("mr", H, st, True) for H in (12, 24) for st in (2.0, 3.0)})   # 目標 = 訊號當下 EMA20
MOM_SPECS = {f"mom_H{H}_s{st}": ("mom", H, st, False) for H in (6, 12, 24, 48) for st in (1.5, 2.0, 3.0)}
PERS = (("IS1", lambda E: E.per == "IS1"), ("IS2", lambda E: E.per == "IS2"), ("IS", lambda E: E.per != "OOS"), ("OOS", lambda E: E.per == "OOS"))


if __name__ == "__main__":
    open("results_part24.txt", "w").close()
    d, A, M, X = load_all()
    RFI_IS = X.loc[(X.index < CUT) & X.rfi.notna(), "rfi"].to_numpy()        # RFI 分位一律用 IS 全部 K 棒
    P("=" * 120)
    P("第二十四部分：回復力切換策略優化（EMA20、M5；IS = 1~5 月選、OOS = 6~10 月驗證）")

    # ---------------------------------------------------------------- 事件與結果（全部參數組合）
    EV = {}
    for d0 in (1.5, 2.0, 2.5):
        for rf in (0.5, 0.75):
            E = event_frame(d, A, M, X, d0, rf)
            EV[(d0, rf)] = add_outcomes(d, E, {**MR_SPECS, **MOM_SPECS})
    qcut = lambda q: np.nanquantile(RFI_IS, [q, 1 - q])

    # ---------------------------------------------------------------- 1. 持倉限制
    P("\n1. 拿掉「同時只持一筆」：原型參數（|D| ≥ 2、回到 1 以內重新武裝、RFI 三分位、兩邊都持有 12 根、停損 2 ATR）")
    E = EV[(2.0, 0.5)]
    lo, hi = qcut(1 / 3)
    mr, mom = E.rfi >= hi, E.rfi <= lo
    T = pd.concat([E[mr].assign(pnl=E["pnl_mr_H12_s2.0"], ex=E["exit_mr_H12_s2.0"], leg="mr"),
                   E[mom].assign(pnl=E["pnl_mom_H12_s2.0"], ex=E["exit_mom_H12_s2.0"], leg="mom")]).sort_values("i").reset_index(drop=True)
    for lab, cap in (("同時 1 筆（原型）", 1), ("同時 ≤ 2 筆", 2), ("同時 ≤ 3 筆", 3), ("不限", None)):
        ok, conc = apply_cap(T.i.to_numpy() + 1, T.ex.to_numpy(), cap)
        S = T[ok]
        P(f"  {lab:14s} IS {fmt(summarize(S[S.per != 'OOS'].pnl, S[S.per != 'OOS'].t))}")
        P(f"  {'':14s} OOS {fmt(summarize(S[S.per == 'OOS'].pnl, S[S.per == 'OOS'].t))}   最多同時 {conc[ok].max()} 筆")
    ok, conc = apply_cap(T.i.to_numpy() + 1, T.ex.to_numpy(), None)
    P(f"  不限時同時持倉數分佈：" + "  ".join(f"{k} 筆 {np.mean(conc == k):.0%}" for k in range(1, conc.max() + 1)))
    P("  註：事件本身已經是『同一段偏離只觸發一次』，所以拿掉持倉限制後增加的是『上一筆還沒出場、價格又產生新的偏離』的單。")
    P("  再放寬重新武裝（D 回到 1.5 以內就可再觸發）：")
    E2 = EV[(2.0, 0.75)]
    mr2, mom2 = E2.rfi >= hi, E2.rfi <= lo
    T2 = pd.concat([E2[mr2].assign(pnl=E2["pnl_mr_H12_s2.0"], ex=E2["exit_mr_H12_s2.0"]), E2[mom2].assign(pnl=E2["pnl_mom_H12_s2.0"], ex=E2["exit_mom_H12_s2.0"])])
    for per in ("IS", "OOS"):
        S = T2[(T2.per != "OOS") if per == "IS" else (T2.per == "OOS")]
        P(f"  {'不限 + 寬鬆武裝':14s} {per:3s} {fmt(summarize(S.pnl, S.t))}")

    # ---------------------------------------------------------------- 2. 參數格子（各邊分開、持倉不限）
    P("\n2. 參數格子（每一邊分開，持倉不限）：|D| 門檻 d0 × 重新武裝 × RFI 分位 × 出場（持有根數 H、停損 ATR 倍數、是否以 EMA 為目標）")
    rows = []
    for (d0, rf), E in EV.items():
        for q in (0.25, 1 / 3, 0.4):
            lo, hi = qcut(q)
            for leg, specs, m in (("mr", MR_SPECS, E.rfi >= hi), ("mom", MOM_SPECS, E.rfi <= lo)):
                for name in specs:
                    r = dict(leg=leg, d0=d0, rearm=rf, q=round(q, 2), exit=name)
                    for lab, f in PERS:
                        sel = E[m & f(E)]
                        s = summarize(sel[f"pnl_{name}"].to_numpy(), sel.t.to_numpy())
                        r.update({f"{k}_{lab}": v for k, v in s.items()})
                    rows.append(r)
    G = pd.DataFrame(rows)
    G.to_csv("part24_grid.csv", index=False)
    for leg, lab in (("mr", "回歸"), ("mom", "動能")):
        g = G[G.leg == leg]
        P(f"  {lab}：{len(g)} 組 | IS PF>1 的比例 {np.mean(g.pf_IS > 1):.0%}（IS1 {np.mean(g.pf_IS1 > 1):.0%}、IS2 {np.mean(g.pf_IS2 > 1):.0%}）"
          f" | OOS PF>1 的比例 {np.mean(g.pf_OOS > 1):.0%}、OOS PF 中位數 {g.pf_OOS.median():.2f}")
    P("  各參數的邊際效果（該參數值下所有組合的 IS / OOS 平均每筆美元）：")
    for leg, lab in (("mr", "回歸"), ("mom", "動能")):
        g = G[G.leg == leg].copy()
        g["H"] = g.exit.str.extract(r"H(\d+)").astype(int); g["stop"] = g.exit.str.extract(r"s([\d.]+)").astype(float)
        g["target"] = g.exit.str.contains("_T_")
        for par in ("d0", "rearm", "q", "H", "stop", "target"):
            P(f"    {lab} {par:6s} " + "  ".join(f"{v}: {gg.avg_IS.mean():+.2f} / {gg.avg_OOS.mean():+.2f}" for v, gg in g.groupby(par)))

    # 選擇規則（只看 IS）：IS1、IS2 都 PF > 1.05 且 IS 筆數 ≥ 60 → 取 IS 總損益/回撤最大
    P("\n  選擇規則（只看 IS）：IS1、IS2 的 PF 都 > 1.05、IS 筆數 ≥ 60，取 IS『總損益 / 最大回撤』最大者")
    pick = {}
    for leg, lab in (("mr", "回歸"), ("mom", "動能")):
        g = G[(G.leg == leg) & (G.pf_IS1 > 1.05) & (G.pf_IS2 > 1.05) & (G.n_IS >= 60)].sort_values("ratio_IS", ascending=False)
        P(f"  {lab}：符合 {len(g)} 組；前 5 名（IS → OOS）")
        for r in g.head(5).itertuples():
            P(f"    d0 {r.d0} 武裝 {r.rearm} q {r.q} {r.exit:14s} IS {r.n_IS:3d}筆 每筆 {r.avg_IS:+5.2f} PF {r.pf_IS:.2f} 總/回撤 {r.ratio_IS:4.1f}"
              f" → OOS {r.n_OOS:3d}筆 每筆 {r.avg_OOS:+5.2f} PF {r.pf_OOS:.2f} 總 {r.total_OOS:+6.0f}")
        if len(g):
            P(f"    符合規則的 {len(g)} 組在 OOS：PF 中位數 {g.pf_OOS.median():.2f}、PF>1 比例 {np.mean(g.pf_OOS > 1):.0%}")
            pick[leg] = g.iloc[0]

    # ---------------------------------------------------------------- 3. 因子調配（在選出的參數上）
    P("\n3. 因子調配：在選出的參數上，每個因子按 IS 三分位切；事先登記規則 = 某一組在 IS1 與 IS2 都是三組中最差、且 IS 平均 < 0 → 刪掉")
    FACT = ["vwap_side", "day_der", "htf", "day_used", "vel", "acc", "er_ratio", "er20", "er10", "leg_bars", "bbw_pct",
            "sig_ratio", "atr_ratio", "dist", "wick_out", "vol_rel", "beyond_pd"]
    NM = {"vwap_side": "VWAP同側距", "day_der": "當天ER順向", "htf": "順高週期", "day_used": "日波幅已用", "vel": "離開速度",
          "acc": "離開加速度", "er_ratio": "ER5/ER20", "er20": "ER20", "er10": "ER10", "leg_bars": "離開根數", "bbw_pct": "帶寬水位",
          "sig_ratio": "σ短/σ長", "atr_ratio": "ATR5/ATR60", "dist": "距離|D|", "wick_out": "外側影線", "vol_rel": "相對量",
          "beyond_pd": "超出前日高低", "sess": "時段", "dow": "星期"}
    final = {}
    for leg, lab in (("mr", "回歸"), ("mom", "動能")):
        if leg not in pick:
            continue
        r = pick[leg]
        E = EV[(r.d0, r.rearm)]
        lo, hi = qcut(r.q)
        m = (E.rfi >= hi) if leg == "mr" else (E.rfi <= lo)
        S = E[m].copy(); col = f"pnl_{r.exit}"
        P(f"\n  --- {lab}（d0 {r.d0}、武裝 {r.rearm}、q {r.q}、{r.exit}）基準：IS {fmt(summarize(S[S.per != 'OOS'][col], S[S.per != 'OOS'].t))}")
        P(f"  {'':12s}                                   OOS {fmt(summarize(S[S.per == 'OOS'][col], S[S.per == 'OOS'].t))}")
        drops = []
        for f in FACT + ["sess", "dow"]:
            if f in ("sess", "dow"):
                g = S[f]
            else:
                e = np.nanquantile(S.loc[S.per != "OOS", f], [1 / 3, 2 / 3])
                g = pd.Series(np.digitize(S[f], e), S.index).where(S[f].notna())
            tab = S.groupby([g, "per"])[col].mean().unstack()
            if not {"IS1", "IS2", "OOS"} <= set(tab.columns):
                continue
            isall = S[S.per != "OOS"].groupby(g[S.per != "OOS"])[col].mean()
            worst1, worst2 = tab.IS1.idxmin(), tab.IS2.idxmin()
            rule = worst1 == worst2 and isall[worst1] < 0
            cells = "  ".join(f"{k}: {tab.loc[k, 'IS1']:+5.1f}/{tab.loc[k, 'IS2']:+5.1f}/{tab.loc[k, 'OOS']:+5.1f}" for k in tab.index)
            P(f"    {NM[f]:10s} {cells}   {'→ 刪除 ' + str(worst1) if rule else ''}")
            if rule:
                drops.append((f, worst1, None if f in ("sess", "dow") else np.nanquantile(S.loc[S.per != "OOS", f], [1 / 3, 2 / 3])))
        keep = pd.Series(True, S.index)
        for f, grp, e in drops:
            g = S[f] if e is None else pd.Series(np.digitize(S[f], e), S.index)
            keep &= g != grp
        P(f"  刪除 {len(drops)} 項後：IS {fmt(summarize(S[keep & (S.per != 'OOS')][col], S[keep & (S.per != 'OOS')].t))}")
        P(f"  {'':10s}         OOS {fmt(summarize(S[keep & (S.per == 'OOS')][col], S[keep & (S.per == 'OOS')].t))}")
        # 隨機刪除對照：OOS 隨機刪掉同樣比例的單 1000 次
        so = S[S.per == "OOS"][col].to_numpy(); kk = keep[S.per == "OOS"].to_numpy()
        rng = np.random.default_rng(1)
        rand = [so[rng.permutation(len(so))[:kk.sum()]].mean() for _ in range(1000)]
        P(f"  OOS 刪除後每筆 {so[kk].mean():+.2f} 勝過隨機刪除的比例 {np.mean(so[kk].mean() > np.array(rand)):.0%}")
        final[leg] = (r, S[keep].assign(pnl=S[keep][col], ex=S[keep][f"exit_{r.exit}"], why=S[keep][f"why_{r.exit}"], leg=lab), drops)

    # ---------------------------------------------------------------- 4. 最終版合併
    P("\n4. 最終版（兩邊合併）")
    T = pd.concat([v[1] for v in final.values()]).sort_values("i").reset_index(drop=True)
    for lab, cap in (("不限", None), ("同時 ≤ 3 筆", 3), ("同時 1 筆", 1)):
        ok, conc = apply_cap(T.i.to_numpy() + 1, T.ex.to_numpy(), cap)
        S = T[ok]
        for per in ("IS", "OOS"):
            s_ = S[(S.per != "OOS") if per == "IS" else (S.per == "OOS")]
            P(f"  {lab:10s} {per:3s} {fmt(summarize(s_.pnl, s_.t))}")
    P("  月份（不限）：" + "  ".join(f"{k.month}月 {g.pnl.sum():+.0f}({len(g)})" for k, g in T.groupby(T.t.dt.to_period("M"))))
    P("  時段（不限，OOS）：" + "  ".join(f"{k} {g.pnl.mean():+.2f}({len(g)})" for k, g in T[T.per == "OOS"].groupby("sess")))
    o = d.open.to_numpy()
    log = pd.DataFrame({"訊號時間": T.t, "進場時間": d.index[T.i + 1], "出場時間": d.index[T.ex] + pd.Timedelta(minutes=5),
                        "類型": T.leg, "方向": np.where((T.leg == "回歸") == (T.D < 0), "多", "空"),
                        "進場價": o[T.i + 1].round(2), "出場原因": T.why, "損益_美元": T.pnl.round(2),
                        "損益_ATR": (T.pnl / T.atr).round(3), "ATR14": T.atr.round(2), "D": T.D.round(2), "RFI": T.rfi.round(3),
                        "時段": T.sess, "期間": np.where(T.per == "OOS", "樣本外", "樣本內")})
    log.to_csv("part24_trades.csv", index=False, encoding="utf-8-sig")
    P(f"  逐筆明細：part24_trades.csv（{len(log)} 筆）")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for lab, col in (("回歸", "tab:green"), ("動能", "tab:red")):
        s_ = T[T.leg == lab]
        ax.plot(s_.t, s_.pnl.cumsum(), color=col, label={"回歸": "mean-reversion leg", "動能": "momentum leg"}[lab])
    ax.plot(T.t, T.pnl.cumsum(), color="tab:blue", lw=2, label="total (no position cap)")
    ax.axvline(CUT, color="k", lw=.8); ax.axhline(0, color="gray", lw=.5)
    ax.set_ylabel("USD per oz (0.01 lot)"); ax.set_title("Part 24 optimised restoring-force switch (left of line = in-sample)")
    ax.legend(); fig.tight_layout(); fig.savefig("part24_equity.png", dpi=110)
