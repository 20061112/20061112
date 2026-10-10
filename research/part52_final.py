"""第五十二部分：很陡順勢 最終版 —— 2023/1 ~ 2026/10 完整交易數據（規格書見 STEEP_SPEC.md）。

規則與第 49j / 50 部分主設定完全相同：M15、L48、EWMA z(cosθ)（span 1000）≤ −2、順勢、1.5 ATR 停損、48 根出場。
輸出：part52_trades.csv、part52_stats.txt、part52_equity.png、part52_monthly.csv、part52_data.json（報告頁用）
"""
import json
import numpy as np
import pandas as pd
from load import load
from part49_friction import features, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_ewm

PATH = "data/XAUUSD_M15_2023_2026.csv"
L, W, THR, SL = 48, 1000, -2.0, 1.5
RISK = 0.005           # 資金模擬：每筆風險 0.5%
CAP0 = 10000.0


def pf(p):
    return p[p > 0].sum() / -p[p < 0].sum()


def streak(b):
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def block(T, name):
    p, R = T.pnl, T.R
    m, t, _ = tst(R.to_numpy())
    eqR = R.cumsum()
    ddR = (eqR - np.maximum.accumulate(np.maximum(eqR, 0))).min()
    return dict(期間=name, 筆數=len(T), 多=int((T.dir == "多").sum()), 空=int((T.dir == "空").sum()),
                勝率=(p > 0).mean(), 每筆R=m, t=t, PF=pf(p), 每筆美元=p.mean(), 總美元=p.sum(), 總R=R.sum(),
                最大回撤R=ddR, 最長連虧=streak((p <= 0).to_numpy()), 停損比例=(T.exit_reason == "停損").mean(),
                最好5筆佔總R=R.nlargest(5).sum() / R.sum() if R.sum() > 0 else np.nan)


def main():
    df = load(PATH)
    F = features(df, L)
    z = z_ewm(F.cos, W)
    valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
    T = backtest(df, F, valid & (z <= THR), L)
    B = backtest(df, F, valid, L)
    tan = np.sqrt(1 / F.cos ** 2 - 1)
    T["theta_deg"] = np.degrees(np.arctan(tan.reindex(T.signal_time).to_numpy()))
    T["z"] = z.reindex(T.signal_time).to_numpy()
    T["hours"] = (T.exit_time - T.entry_time).dt.total_seconds() / 3600
    T["spread_cost"] = (df.spread * 0.01).reindex(T.entry_time).to_numpy()
    # 固定風險資金曲線（不複利）
    T["acct_pnl"] = T.R * RISK * CAP0
    T["acct_eq"] = CAP0 + T.acct_pnl.cumsum()
    T.to_csv("part52_trades.csv", index=False, float_format="%.4f")

    y = T.entry_time.dt.year
    rows = [block(T[y == k], str(k)) for k in (2023, 2024, 2025, 2026)]
    rows += [block(T[y <= 2025], "2023~25 樣本外"), block(T, "全部 2023~26")]
    S = pd.DataFrame(rows).set_index("期間")
    by = B.entry_time.dt.year
    S["基準每筆R"] = [B[by == k].R.mean() for k in (2023, 2024, 2025, 2026)] + [B[by <= 2025].R.mean(), B.R.mean()]
    S["超額"] = S.每筆R - S.基準每筆R

    lines = ["第五十二部分：很陡順勢 最終版 完整交易數據", f"資料 {df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}（M15，伺服器時間）",
             "2026/1 ~ 10 是研究期間（第 49 部分），2023 ~ 2025 是樣本外", ""]
    lines.append(S.round(3).to_string())
    p, R = T.pnl, T.R
    lines += ["", "===== 全期細節 =====",
              f"平均賺 / 平均賠：{R[p > 0].mean():+.2f}R / {R[p <= 0].mean():+.2f}R（賺賠比 {R[p > 0].mean() / -R[p <= 0].mean():.2f}）",
              f"R 中位數 {R.median():+.2f}；最大單筆 {R.max():+.1f}R；最小 {R.min():+.2f}R",
              f"出場：停損 {np.mean(T.exit_reason == '停損'):.0%}（平均 {R[T.exit_reason == '停損'].mean():+.2f}R）／時間到 {np.mean(T.exit_reason == '時間'):.0%}（平均 {R[T.exit_reason == '時間'].mean():+.2f}R）",
              f"持倉：平均 {T.hours.mean():.1f} 小時（中位數 {T.hours.median():.1f}）；贏單 {T.hours[p > 0].mean():.1f}、輸單 {T.hours[p <= 0].mean():.1f}",
              f"跨週末持倉：{int((T.hours > 13).sum())} 筆（平均 {R[T.hours > 13].mean():+.2f}R），跳空風險見規格書",
              f"停損距離中位數：{T.risk.median():.2f} 美元（2023 {T.risk[y == 2023].median():.1f} → 2026 {T.risk[y == 2026].median():.1f}）",
              f"點差成本中位數：{(T.spread_cost / T.risk).median():.1%} R",
              f"MFE 中位數 {T.mfe_R.median():.2f}R；輸單曾浮盈 ≥ 1R：{np.mean(T.mfe_R[p <= 0] >= 1):.0%}",
              f"訊號角度 θ：中位數 {T.theta_deg.median():.1f}°（{T.theta_deg.min():.0f}° ~ {T.theta_deg.max():.0f}°）",
              f"在場時間：{T.hours.sum() / ((df.index.max() - df.index.min()).days * 24 * 5 / 7):.0%}（以交易時段估）",
              f"每週平均筆數：{len(T) / ((df.index.max() - df.index.min()).days / 7):.1f}"]
    for d in ("多", "空"):
        s = T[T.dir == d]
        m, t, _ = tst(s.R.to_numpy())
        lines.append(f"{d}單：n={len(s)} 勝率 {(s.pnl > 0).mean():.0%} {m:+.3f}R(t={t:+.1f}) PF {pf(s.pnl):.2f}")
    lines += ["", "進場星期："]
    for k, s in T.groupby(T.entry_time.dt.dayofweek):
        lines.append(f"  週{'一二三四五六日'[k]} n={len(s):3d} 勝率 {(s.pnl > 0).mean():.0%} 每筆 {s.R.mean():+.3f}R")
    lines += ["", "進場時段（伺服器時間）："]
    hb = pd.cut(T.entry_time.dt.hour, [-1, 7, 13, 18, 23], labels=["0-7", "8-13", "14-18", "19-23"])
    for k, s in T.groupby(hb, observed=True):
        lines.append(f"  {k:6s} n={len(s):3d} 勝率 {(s.pnl > 0).mean():.0%} 每筆 {s.R.mean():+.3f}R")
    # 每月 R
    M = T.groupby(T.entry_time.dt.to_period("M")).agg(筆數=("R", "size"), 總R=("R", "sum"), 勝率=("pnl", lambda v: (v > 0).mean()))
    M.to_csv("part52_monthly.csv")
    lines += ["", f"每月總 R：正的月份 {(M.總R > 0).mean():.0%}（{(M.總R > 0).sum()} / {len(M)}），最好 {M.總R.max():+.1f}R，最差 {M.總R.min():+.1f}R",
              f"連續虧損月份最長：{streak((M.總R <= 0).to_numpy())} 個月"]
    # 資金
    eq = T.acct_eq
    dd = eq / eq.cummax().clip(lower=CAP0) - 1
    lines += ["", f"===== 資金模擬：1 萬美元、每筆風險 {RISK:.1%}、不複利 =====",
              f"最終 {eq.iloc[-1]:,.0f}（{eq.iloc[-1] / CAP0 - 1:+.0%}），最大回撤 {dd.min():.1%}"]
    for k in (2023, 2024, 2025, 2026):
        s = T[y == k]
        lines.append(f"  {k}：{s.acct_pnl.sum() / CAP0:+.1%}")
    for r in (0.0025, 0.01):
        e = CAP0 + (T.R * r * CAP0).cumsum()
        d = (e / e.cummax().clip(lower=CAP0) - 1).min()
        lines.append(f"  每筆風險 {r:.2%}：總報酬 {e.iloc[-1] / CAP0 - 1:+.0%}，最大回撤 {d:.1%}")
    txt = "\n".join(lines)
    print(txt)
    with open("part52_stats.txt", "w") as fh:
        fh.write(txt + "\n")

    # 報告頁資料
    data = dict(
        summary=S.reset_index().round(4).to_dict(orient="records"),
        eqR=[[t.strftime("%Y-%m-%d"), round(v, 2)] for t, v in zip(T.exit_time, T.R.cumsum())],
        baseR=[[t.strftime("%Y-%m-%d"), round(v * len(T) / len(B), 2)] for t, v in zip(B.exit_time[::4], B.R.cumsum()[::4])],
        acct=[[t.strftime("%Y-%m-%d"), round(v, 0)] for t, v in zip(T.exit_time, T.acct_eq)],
        monthly=[[str(k), int(r.筆數), round(r.總R, 2)] for k, r in M.iterrows()],
        trades=[dict(e=r.entry_time.strftime("%Y-%m-%d %H:%M"), x=r.exit_time.strftime("%m-%d %H:%M"), d=r.dir,
                     ep=round(r.entry, 2), xp=round(r.exit, 2), rk=round(r.risk, 2), th=round(r.theta_deg, 1),
                     why=r.exit_reason, h=round(r.hours, 1), p=round(r.pnl, 2), R=round(r.R, 2)) for r in T.itertuples()],
        lines=lines)
    json.dump(data, open("part52_data.json", "w"), ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(T.exit_time, T.R.cumsum(), color="#2a6fdb", lw=1.6, label=f"很陡順勢（{len(T)} 筆）")
    ax[0].plot(B.exit_time, B.R.cumsum() * len(T) / len(B), color="#999", lw=1, label="全部順勢（按筆數比例縮放）")
    ax[0].axvline(pd.Timestamp("2026-01-01"), color="#c33", ls="--", lw=0.8)
    ax[0].set_title("累積 R（2023 ~ 2025 樣本外，2026 研究期間）")
    ax[0].legend(loc="upper left")
    ax[0].grid(alpha=0.3)
    ax[1].plot(T.exit_time, dd * 100, color="#d9534f")
    ax[1].fill_between(T.exit_time, dd * 100, 0, color="#d9534f", alpha=0.3)
    ax[1].set_title(f"帳戶回撤 %（每筆風險 {RISK:.1%}）")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("part52_equity.png", dpi=110)


if __name__ == "__main__":
    main()
