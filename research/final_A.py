"""定案：平衡版 A —— 逐筆明細、逐分鐘浮動損益（帳戶層級最大浮虧）、持倉時間、圖表、實盤門檻。

規則見 FINAL_STRATEGY_A.md。門檻用滾動前推（每月月初只用過去資料重算），報告期間 2026/3 ~ 10/8。
輸出：
  final_A_trades.csv      逐筆明細
  final_A_stats.txt       統計
  final_A_*.png           圖表
  final_A_live.json       實盤用門檻（用到 10/8 為止的全部資料計算）
"""
import json
import numpy as np
import pandas as pd
from part14_deep import ARR, DAYS
from part17_balance import gen

Q_IB, Q_TAIL, Q_EXT = 0.4, 0.9, 0.9
CFG = dict(confirm="close5", buf=0.05, hours=(2, 10), step=30, monday=False)


def walk_forward(X):
    X = X.copy(); X["ym"] = X.day.dt.to_period("M")
    parts, used = [], []
    for m in sorted(X.ym.unique())[2:]:
        past, cur = X[X.ym < m], X[X.ym == m]
        ti, tt, te = past.ib_atr.quantile(Q_IB), past["tail"].quantile(Q_TAIL), past.ext_letter.quantile(Q_EXT)
        used.append(dict(month=str(m), thr_ib=ti, thr_tail=tt, thr_ext=te))
        keep = ~(cur.aligned & (cur.ib_atr < ti)) & (cur["tail"] <= tt) & (cur.ext_letter.fillna(0) <= te)
        parts.append(cur[keep])
    Y = pd.concat(parts).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
    return Y.sort_values("t_in").reset_index(drop=True), pd.DataFrame(used)


def paths(T):
    """每筆的逐分鐘浮動損益（最差：多單用 low、空單用 high），以及出場時的實現損益。"""
    rows, extra = [], []
    for n, r in enumerate(T.itertuples()):
        t, O, H, L, C, S = ARR[r.day]
        j = int(np.searchsorted(t, r.exit_time))
        k0 = r.i
        if j > k0:
            worst = r.side * ((L if r.side == 1 else H)[k0:j] - r.lvl) - r.spread
            rows.append(pd.Series(worst, index=t[k0:j], name=n))
            k_mae = k0 + int(np.argmin(worst))
            t_mae = (t[k_mae] - r.t_in).total_seconds() / 60
        else:
            t_mae = 0.0
        extra.append(t_mae)
    return rows, extra


def main():
    X = gen(**CFG)
    T, used = walk_forward(X)
    rows, t_mae = paths(T)
    T["mae_min"] = t_mae
    T["mae_R"] = T.mae / T.risk; T["mfe_R"] = T.mfe / T.risk
    T["ib_start"] = T.h.map(lambda h: f"{int(h):02d}:{int(round((h % 1) * 60)):02d}")
    T["dir"] = T.side.map({1: "多", -1: "空"})
    T["stop_px"] = np.where(T.side == 1, T.ibl, T.ibh)
    T["lots_0.5pct_10k"] = (10000 * 0.005 / (T.risk * 100)).clip(lower=0.01).round(2)

    # ---- 帳戶層級逐分鐘權益（每筆 1 盎司）----
    U = pd.concat(rows, axis=1).sort_index()
    open_pnl = U.sum(axis=1, min_count=1).fillna(0)
    n_open = U.notna().sum(axis=1)
    realized = T.set_index("exit_time").pnl.groupby(level=0).sum().sort_index()
    idx = open_pnl.index.union(realized.index)
    real_cum = realized.reindex(idx, fill_value=0).cumsum()
    eq_mtm = real_cum + open_pnl.reindex(idx, fill_value=0)
    dd_mtm = eq_mtm - eq_mtm.cummax()
    eq_closed = realized.cumsum(); dd_closed = eq_closed - eq_closed.cummax()

    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    lose = (p <= 0).astype(int); streak = lose.groupby((lose != lose.shift()).cumsum()).sum().max()
    tdays = [d for d in DAYS if T.day.min() <= d <= T.day.max()]
    daily = pd.Series(0.0, index=tdays).add(T.groupby("day").pnl.sum(), fill_value=0)
    daily_nomon = daily[[d.dayofweek != 0 for d in daily.index]]
    S = {}
    S["期間"] = f"{T.day.min().date()} ~ {T.day.max().date()}（滾動前推，每月只用過去資料重算門檻）"
    S["交易筆數"] = len(T)
    S["交易日（不含週一）"] = len(daily_nomon)
    S["有交易的日子"] = int(T.day.nunique())
    S["每個交易日平均筆數"] = round(len(T) / len(daily_nomon), 2)
    S["有交易日的平均筆數"] = round(len(T) / T.day.nunique(), 2)
    S["多 / 空"] = f"{(T.side == 1).sum()} / {(T.side == -1).sum()}"
    S["勝率"] = f"{len(w) / len(p) * 100:.1f}%"
    S["平均賺 / 平均賠"] = f"{w.mean():+.2f} / {l.mean():+.2f}"
    S["賺賠比"] = round(w.mean() / -l.mean(), 2)
    S["每筆期望（美元/盎司）"] = round(p.mean(), 2)
    S["每筆期望（R）"] = f"{T.R.mean():+.2f}（中位數 {T.R.median():+.2f}）"
    S["PF"] = round(w.sum() / -l.sum(), 2)
    S["總損益（每筆 1 盎司）"] = round(p.sum(), 0)
    S["最大回撤（平倉後）"] = round(dd_closed.min(), 0)
    S["最大回撤（含浮動，逐分鐘）"] = round(dd_mtm.min(), 0)
    S["最大回撤發生時間（含浮動）"] = str(dd_mtm.idxmin())
    S["總損益 / 最大回撤（含浮動）"] = round(p.sum() / -dd_mtm.min(), 1)
    S["日 Sharpe（年化，不含週一）"] = round(daily_nomon.mean() / daily_nomon.std() * np.sqrt(250), 2)
    S["正報酬日 / 零交易日"] = f"{(daily_nomon > 0).mean() * 100:.0f}% / {(daily_nomon == 0).mean() * 100:.0f}%"
    S["最差單日 / 最好單日"] = f"{daily.min():+.0f} / {daily.max():+.0f}"
    S["最長連虧 / 連勝（筆）"] = f"{streak} / {(p > 0).astype(int).groupby(((p > 0) != (p > 0).shift()).cumsum()).sum().max()}"
    S["單筆最大浮虧（美元/盎司）"] = f"{T.mae.max():.1f}（{T.mae_R.max():.2f}R，{T.loc[T.mae.idxmax(), 'entry_time']}）"
    S["單筆浮虧 中位數 / 90% / 95%"] = f"{T.mae.median():.1f} / {T.mae.quantile(.9):.1f} / {T.mae.quantile(.95):.1f}"
    S["單筆浮虧（R）中位數 / 贏單中位數"] = f"{T.mae_R.median():.2f}R / {T[T.pnl > 0].mae_R.median():.2f}R"
    S["帳戶同時浮虧最大（所有持倉加總）"] = f"{open_pnl.min():.1f}（{open_pnl.idxmin()}，當時持有 {int(n_open[open_pnl.idxmin()])} 筆）"
    S["同時持倉 最多 / 平均（有持倉時）"] = f"{int(n_open.max())} / {n_open[n_open > 0].mean():.2f}"
    S["單筆最大浮盈"] = f"{T.mfe.max():.1f}（{T.mfe_R.max():.1f}R）"
    S["輸單中曾浮盈 ≥ 1R 的比例"] = f"{(T[T.pnl <= 0].mfe_R >= 1).mean() * 100:.0f}%"
    S["平均持倉時間"] = f"{T.hold_h.mean():.2f} 小時（中位數 {T.hold_h.median():.2f}）"
    S["停損單 平均持倉"] = f"{T[T.exit_reason == '停損'].hold_h.mean():.2f} 小時（中位數 {T[T.exit_reason == '停損'].hold_h.median():.2f}）"
    S["收盤出場 平均持倉"] = f"{T[T.exit_reason == '收盤'].hold_h.mean():.2f} 小時"
    S["贏單 / 輸單 平均持倉"] = f"{T[T.pnl > 0].hold_h.mean():.2f} / {T[T.pnl <= 0].hold_h.mean():.2f} 小時"
    S["出場：停損 / 收盤"] = f"{(T.exit_reason == '停損').mean() * 100:.0f}% / {(T.exit_reason == '收盤').mean() * 100:.0f}%"
    S["最大浮虧出現在進場後（中位數）"] = f"{T.mae_min.median():.0f} 分鐘"
    S["停損距離 中位數（美元）"] = f"{T.risk.median():.1f}（{(T.risk / T.atr).median():.2f} ATR），範圍 {T.risk.min():.1f}~{T.risk.max():.1f}"
    # 帳戶百分比（固定 1 萬美元基準、每筆風險 r%、不複利）：含浮動的最大回撤
    for r in (0.25, 0.5, 1.0):
        wgt = 10000 * r / 100 / T.risk                       # 每筆盎司數
        op = (U * wgt.values).sum(axis=1, min_count=1).fillna(0)
        rz = (T.pnl * wgt).groupby(T.exit_time).sum().sort_index()
        ix = op.index.union(rz.index)
        e = 10000 + rz.reindex(ix, fill_value=0).cumsum() + op.reindex(ix, fill_value=0)
        ec = 10000 + rz.cumsum()
        S[f"每筆風險 {r}%（1 萬美元、不複利）"] = (f"總報酬 {(ec.iloc[-1] / 1e4 - 1) * 100:+.0f}%；最大回撤 平倉 {((ec - ec.cummax()) / ec.cummax()).min() * 100:.1f}% "
                                         f"/ 含浮動 {((e - e.cummax()) / e.cummax()).min() * 100:.1f}%；帳戶同時最大浮虧 {op.min() / 100:.1f}%")
    top = T.groupby("day").pnl.sum().sort_values(ascending=False)
    S["最好 5 天占總損益"] = f"{top.head(5).sum() / p.sum() * 100:.0f}%；拿掉最好 10 天剩 {top.iloc[10:].sum():+.0f}"

    # 分組表
    def grp(g):
        a = T.groupby(g).agg(筆數=("pnl", "size"), 勝率=("pnl", lambda x: (x > 0).mean() * 100), 每筆=("pnl", "mean"),
                             總計=("pnl", "sum"), 平均持倉h=("hold_h", "mean"), 平均浮虧=("mae", "mean"))
        a["PF"] = T.groupby(g).pnl.apply(lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9))
        return a.round(2)
    G = {"月份": grp(T.day.dt.strftime("%Y-%m")), "IB 起點": grp("ib_start"), "方向": grp("dir"),
         "星期": grp(T.day.dt.dayofweek.map(dict(enumerate("一二三四五")))), "出場原因": grp("exit_reason")}
    hb = pd.cut(T.hold_h, [0, 0.5, 1, 2, 4, 8, 12, 16, 24], right=False)
    G["持倉時間"] = grp(hb).rename(index=lambda iv: f"{iv.left:g}~{iv.right:g}h")

    # 實盤門檻：用所有資料
    live = dict(thr_ib_atr=float(X.ib_atr.quantile(Q_IB)), thr_tail_atr=float(X["tail"].quantile(Q_TAIL)),
                thr_ext_letter=float(X.ext_letter.quantile(Q_EXT)), computed_on=f"{X.day.min().date()}~{X.day.max().date()}",
                note="每月月初用到上月底為止的全部訊號重算（IB/ATR 40%、tail 90%、ext_letter 90% 分位）")
    json.dump(live, open("final_A_live.json", "w"), ensure_ascii=False, indent=2)

    cols = ["day", "dir", "ib_start", "ibh", "ibl", "entry_time", "lvl", "stop_px", "risk", "exit_time", "exit_px", "exit_reason",
            "pnl", "R", "mae", "mae_R", "mae_min", "mfe", "mfe_R", "hold_h", "atr", "ib_atr", "aligned", "tail", "ext_letter",
            "spread", "lots_0.5pct_10k"]
    out = T[cols].rename(columns={"day": "交易日", "dir": "方向", "ib_start": "IB起點", "ibh": "IB高", "ibl": "IB低",
                                  "entry_time": "進場時間", "lvl": "進場價", "stop_px": "停損價", "risk": "停損距離",
                                  "exit_time": "出場時間", "exit_px": "出場價", "exit_reason": "出場原因", "pnl": "損益_美元每盎司",
                                  "R": "損益_R", "mae": "最大浮虧", "mae_R": "最大浮虧_R", "mae_min": "最大浮虧出現_分鐘",
                                  "mfe": "最大浮盈", "mfe_R": "最大浮盈_R", "hold_h": "持倉_小時", "atr": "ATR10",
                                  "ib_atr": "IB除ATR", "aligned": "順前日方向", "tail": "反向尾巴_ATR", "ext_letter": "反向極值字母位置",
                                  "spread": "點差", "lots_0.5pct_10k": "1萬美元0.5%風險手數"})
    out.to_csv("final_A_trades.csv", index=False, float_format="%.3f", encoding="utf-8-sig")
    used.to_csv("final_A_thresholds_by_month.csv", index=False, float_format="%.4f")

    with open("final_A_stats.txt", "w") as f:
        for k, v in S.items():
            f.write(f"{k}: {v}\n")
        for k, v in G.items():
            f.write(f"\n[{k}]\n{v.to_string()}\n")
        f.write("\n[每月門檻]\n" + used.round(4).to_string(index=False) + "\n")

    charts(T, eq_mtm, dd_mtm, eq_closed, n_open, G)
    return T, S, G, used, live


def charts(T, eq_mtm, dd_mtm, eq_closed, n_open, G):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
    BLUE, ORANGE, GREEN, RED, GRAY = "#3b6ea8", "#c9824a", "#2e8b57", "#c0392b", "#888888"

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    em = eq_mtm.resample("15min").last().dropna()
    ax[0].plot(em.index, em.values, color=BLUE, lw=1, label="equity incl. open positions (mark-to-market)")
    ax[0].plot(eq_closed.index, eq_closed.values, color=ORANGE, lw=1.3, drawstyle="steps-post", label="closed-trade equity")
    ax[0].set_ylabel("USD per oz (1 oz per trade)"); ax[0].legend(loc="upper left"); ax[0].set_title("Version A — equity (walk-forward, Mar–Oct 2026)")
    dm = dd_mtm.resample("15min").min().dropna()
    ax[1].fill_between(dm.index, dm.values, 0, color=RED, alpha=.5); ax[1].set_ylabel("drawdown incl. open")
    plt.tight_layout(); plt.savefig("final_A_equity.png", dpi=110); plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].hist(T.mae, bins=40, color=RED, alpha=.8); ax[0, 0].set_title("max adverse excursion per trade (USD/oz)")
    ax[0, 0].axvline(T.mae.median(), color="k", ls="--", lw=1); ax[0, 0].text(T.mae.median(), ax[0, 0].get_ylim()[1] * .9, f" median {T.mae.median():.1f}")
    c = np.where(T.pnl > 0, GREEN, RED)
    ax[0, 1].scatter(T.mae_R, T.mfe_R, c=c, s=9, alpha=.55); ax[0, 1].set_yscale("symlog")
    ax[0, 1].set_xlabel("MAE / risk"); ax[0, 1].set_ylabel("MFE / risk"); ax[0, 1].set_title("MAE vs MFE (green = winner)")
    ax[1, 0].hist(T.R.clip(-1.3, 8), bins=45, color=BLUE); ax[1, 0].set_title("R multiple per trade"); ax[1, 0].set_xlabel("R")
    ax[1, 1].hist(T.mae_min.clip(0, 600), bins=40, color=GRAY); ax[1, 1].set_title("minutes from entry to worst point"); ax[1, 1].set_xlabel("minutes")
    plt.tight_layout(); plt.savefig("final_A_risk.png", dpi=110); plt.close()

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].hist([T[T.exit_reason == "停損"].hold_h, T[T.exit_reason == "收盤"].hold_h], bins=24, range=(0, 22), stacked=True,
                  color=[RED, GREEN], label=["stopped", "closed at day end"])
    ax[0, 0].legend(); ax[0, 0].set_title("holding time (hours)")
    no = n_open[n_open > 0]
    ax[0, 1].hist(no, bins=range(1, int(no.max()) + 2), color=BLUE, align="left", density=True)
    ax[0, 1].set_title("open positions at the same time (share of minutes with ≥1 open)")
    mo = G["月份"]
    ax[1, 0].bar(mo.index, mo["總計"], color=[GREEN if v > 0 else RED for v in mo["總計"]])
    ax[1, 0].set_title("monthly PnL (USD/oz)"); ax[1, 0].tick_params(axis="x", rotation=45)
    ib = G["IB 起點"]
    ax[1, 1].bar(ib.index, ib["每筆"], color=BLUE); ax[1, 1].set_title("avg PnL per trade by IB start (broker time)")
    ax[1, 1].tick_params(axis="x", rotation=90)
    plt.tight_layout(); plt.savefig("final_A_breakdown.png", dpi=110); plt.close()

    # 範例日：最好與最差的一天的價格與進出場
    daily = T.groupby("day").pnl.sum()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for a, d, lab in [(ax[0], daily.idxmax(), "best day"), (ax[1], daily.idxmin(), "worst day")]:
        t, O, H, L, C, S = ARR[d]
        a.plot(t, C, color=GRAY, lw=.8)
        for r in T[T.day == d].itertuples():
            col = GREEN if r.side == 1 else RED
            a.plot([r.entry_time, r.exit_time], [r.lvl, r.exit_px], color=col, lw=1.2, alpha=.8)
            a.scatter([r.entry_time], [r.lvl], color=col, marker="^" if r.side == 1 else "v", s=30, zorder=3)
        a.set_title(f"{lab}: {d.date()}  PnL {daily[d]:+.0f} USD/oz, {int((T.day == d).sum())} trades")
        a.tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig("final_A_days.png", dpi=110); plt.close()


if __name__ == "__main__":
    T, S, G, used, live = main()
    for k, v in S.items():
        print(f"{k}: {v}")
    for k, v in G.items():
        print(f"\n[{k}]\n{v.to_string()}")
    print(used.round(4).to_string(index=False)); print(live)
