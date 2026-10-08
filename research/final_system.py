"""目前最佳系統（彙整第 10~16 部分）—— 完整規則、逐筆明細、統計報告、滾動前推。

規則（broker 時間 = 紐約 + 7；交易日 01:00~23:59）
 1. 不做星期一。
 2. 每天 02:00、03:00 … 10:00 各建一個 IB = 該整點起 60 分鐘的最高 / 最低。
 3. IB 結束後 3 小時內，第一根「M5 收盤」站上 IB 高 + 0.05×ATR10 → 做多；跌破 IB 低 − 0.05×ATR10 → 做空；
    以該 M5 收盤價進場。每個 IB 最多一筆，不同 IB 可同時持有。
 4. 不做：突破方向 = 前一交易日 (收−開) 方向，且 IB 區間 < thr_ib × ATR10（小 IB 順勢假突破）。
 5. 不做：當天發展中 TPO（15 分字母）反方向極端的單印尾巴 > thr_tail × ATR10（剛發生急速反轉）。
 6. 不做：反方向的當天極值是由「最近」的字母做出來的（ext_letter > thr_ext，極值很新）。
 7. 停損：IB 另一側。出場：停損或當天最後一根 K 棒收盤。成本：進場 K 棒點差。
 8. 同一根 K 棒、同方向只開一筆（多個 IB 同時觸發時保留最早的 IB；避免隱形加倍）。
門檻 thr_ib / thr_tail / thr_ext 是分位數：IB/ATR 40%、tail 80%、ext_letter 80%。
  固定版：用 1~5 月的分位數；滾動版：每月月初用「之前所有資料」重算（真正可實盤的做法）。
單位：美元/盎司 = 下 0.01 手的美元；R = 損益 / 初始風險（進場價到停損的距離）。
"""
import numpy as np
import pandas as pd
from part14_deep import ARR, D, DAYS, SPLIT
from part15_letters import features as basic_features
from part16_adv_letters import adv

BUF, IB_MIN, WIN, HOURS = 0.05, 60, 3, range(2, 11)


def raw_signals():
    """規則 1~3 的所有訊號，附上規則 4~6 要用的特徵（不在這裡篩）。"""
    out = []
    for di, day in enumerate(DAYS):
        if di == 0 or day.dayofweek == 0:
            continue
        atr = D.atr10.iloc[di]
        if not np.isfinite(atr):
            continue
        prev = D.iloc[di - 1]; pdir = np.sign(prev.close - prev.open)
        t, O, H, L, C, S = ARR[day]
        mins = (t - day).total_seconds().values // 60
        for h in HOURS:
            s0 = h * 60
            m = (mins >= s0) & (mins < s0 + IB_MIN)
            if m.sum() < IB_MIN * 0.8:
                continue
            ibh, ibl = H[m].max(), L[m].min()
            up_l, dn_l = ibh + BUF * atr, ibl - BUF * atr
            w = np.where((mins >= s0 + IB_MIN) & (mins < s0 + IB_MIN + WIN * 60))[0]
            w5 = w[(mins[w] + 1) % 5 == 0]
            up = w5[C[w5] > up_l]; dn = w5[C[w5] < dn_l]
            if len(up) and (not len(dn) or up[0] < dn[0]): side, i = 1, up[0]
            elif len(dn): side, i = -1, dn[0]
            else: continue
            out.append(dict(day=day, h=h, i=min(i + 1, len(C) - 1), i_sig=i, side=side, lvl=C[i], ibh=ibh, ibl=ibl,
                            rng=ibh - ibl, atr=atr, ib_atr=(ibh - ibl) / atr, aligned=bool(side == pdir),
                            t_in=t[i], spread=S[i]))
    s = pd.DataFrame(out)
    s["tail"] = [basic_features(r, 15)["tail"] for r in s.itertuples()]
    s["ext_letter"] = [adv(r).get("ext_letter", np.nan) for r in s.itertuples()]
    return s


def simulate(s):
    """逐筆出場，附 MAE / MFE / 持有時間。"""
    rows = []
    for r in s.itertuples():
        t, O, H, L, C, S = ARR[r.day]
        side, lvl = r.side, r.lvl
        stop = r.ibl if side == 1 else r.ibh
        risk = side * (lvl - stop)
        px, j, why = C[-1], len(C) - 1, "收盤"
        mae = mfe = 0.0
        for k in range(r.i, len(C)):
            hit = L[k] <= stop if side == 1 else H[k] >= stop
            if hit:
                px = min(O[k], stop) if side == 1 else max(O[k], stop); j, why = k, "停損"
                mae = max(mae, risk); break
            mae = max(mae, side * (lvl - (L[k] if side == 1 else H[k])))
            mfe = max(mfe, side * ((H[k] if side == 1 else L[k]) - lvl))
        pnl = side * (px - lvl) - r.spread
        rows.append(dict(entry_time=r.t_in, exit_time=t[j], exit_px=px, exit_reason=why, risk=risk, pnl=pnl,
                         R=pnl / risk if risk > 0 else np.nan, mae=mae, mfe=mfe, hold_h=(t[j] - r.t_in).total_seconds() / 3600))
    return pd.concat([s.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def apply_rules(X, q_ib, q_tail, q_ext):
    Y = X[~(X.aligned & (X.ib_atr < q_ib)) & (X["tail"] <= q_tail) & (X.ext_letter.fillna(0) <= q_ext)]
    return Y.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])


def thresholds(H):
    return H.ib_atr.quantile(0.4), H["tail"].quantile(0.8), H.ext_letter.quantile(0.8)


def walk_forward(X):
    """每月月初，用之前所有訊號重算三個門檻（擴張視窗），只套用在當月。從第 3 個月開始。"""
    X = X.copy(); X["ym"] = X.day.dt.to_period("M")
    months = sorted(X.ym.unique()); parts, used = [], []
    for m in months[2:]:
        past = X[X.ym < m]; cur = X[X.ym == m]
        q = thresholds(past); used.append((str(m), *q))
        parts.append(apply_rules(cur, *q))
    return pd.concat(parts), pd.DataFrame(used, columns=["月", "thr_ib", "thr_tail", "thr_ext"])


def report(T, P, title):
    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    nd = T.day.nunique(); td = len(set(DAYS) & set(T.day.unique()))
    T = T.sort_values("exit_time"); eq = T.pnl.cumsum(); dd = eq - eq.cummax()
    daily = T.groupby("day").pnl.sum()
    all_days = pd.Series(0.0, index=[d for d in DAYS if d >= T.day.min() and d <= T.day.max() and d.dayofweek != 0])
    daily_full = all_days.add(daily, fill_value=0)
    sharpe = daily_full.mean() / daily_full.std() * np.sqrt(250)
    lose = (p <= 0).astype(int); streak = lose.groupby((lose != lose.shift()).cumsum()).sum().max()
    win_s = (p > 0).astype(int); wstreak = win_s.groupby((win_s != win_s.shift()).cumsum()).sum().max()
    # 同時持倉
    ev = sorted([(a, 1) for a in T.entry_time] + [(b, -1) for b in T.exit_time], key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, e in ev:
        cur += e; mx = max(mx, cur)
    P(f"\n{'=' * 100}\n{title}")
    P(f"  交易 {len(T)} 筆；有交易的天 {nd}（{len(T) / nd:.1f} 筆/天）；涵蓋交易日（不含週一）{len(daily_full)}")
    P(f"  勝率 {len(w) / len(p) * 100:.1f}%  平均賺 {w.mean():+.2f}  平均賠 {l.mean():+.2f}  賺賠比 {w.mean() / -l.mean():.2f}")
    P(f"  每筆期望 {p.mean():+.2f} 美元/盎司（{T.R.mean():+.2f} R，中位數 {T.R.median():+.2f} R）  PF {w.sum() / -l.sum():.2f}")
    P(f"  總計 {p.sum():+.0f}  最大回撤 {dd.min():.0f}（= {-dd.min() / T.risk.median():.1f} 筆中位數風險）  總計/回撤 {p.sum() / -dd.min():.1f}")
    P(f"  日報酬 Sharpe（年化，含無交易日）{sharpe:.2f}；正報酬日 {(daily_full > 0).mean() * 100:.0f}%  零交易日 {(daily_full == 0).mean() * 100:.0f}%")
    P(f"  最長連虧 {streak} 筆；最長連勝 {wstreak} 筆；同時持倉最多 {mx} 筆")
    P(f"  初始風險（停損距離）中位數 {T.risk.median():.1f} 美元（{(T.risk / T.atr).median():.2f} ATR），範圍 {T.risk.min():.1f}~{T.risk.max():.1f}")
    P(f"  持有時間 中位數 {T.hold_h.median():.1f} 小時；停損出場 {(T.exit_reason == '停損').mean() * 100:.0f}%（平均持有 "
      f"{T[T.exit_reason == '停損'].hold_h.mean():.1f}h），收盤出場 {(T.exit_reason == '收盤').mean() * 100:.0f}%（平均 {T[T.exit_reason == '收盤'].pnl.mean():+.1f}）")
    win_T = T[T.pnl > 0]; los_T = T[T.pnl <= 0]
    P(f"  MFE（最大浮盈）中位數：贏單 {win_T.mfe.median():.1f}，輸單 {los_T.mfe.median():.1f}（輸單中 MFE ≥ 1R 的比例 "
      f"{(los_T.mfe >= los_T.risk).mean() * 100:.0f}%）")
    P(f"  MAE（最大浮虧）中位數：贏單 {win_T.mae.median():.1f}（{(win_T.mae / win_T.risk).median():.2f}R）")
    P(f"  R 分佈：≤−1R {(T.R <= -0.99).mean() * 100:.0f}%｜−1~0 {((T.R > -0.99) & (T.R <= 0)).mean() * 100:.0f}%｜0~1 {((T.R > 0) & (T.R <= 1)).mean() * 100:.0f}%"
      f"｜1~3 {((T.R > 1) & (T.R <= 3)).mean() * 100:.0f}%｜>3R {(T.R > 3).mean() * 100:.0f}%；最大單筆 {T.R.max():+.1f}R / {p.max():+.0f}")
    top = daily.sort_values(ascending=False)
    P(f"  獲利集中：最好 5 天 {top.head(5).sum():+.0f}（{top.head(5).sum() / p.sum() * 100:.0f}%）；拿掉最好 10 天剩 {top.iloc[10:].sum():+.0f}")
    for lab, g in [("多/空", T.side.map({1: "多", -1: "空"})), ("時段", T.h), ("星期", T.day.dt.dayofweek.map(dict(enumerate("一二三四五")))),
                   ("月份", T.day.dt.strftime("%m"))]:
        a = T.groupby(g).pnl.agg(["count", "mean", "sum"])
        pf = T.groupby(g).pnl.apply(lambda x: x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9))
        P(f"  依{lab}：" + "  ".join(f"{k}: {int(v['count'])}筆 {v['mean']:+.1f} PF{pf[k]:.2f}" for k, v in a.iterrows()))
    wk = T.groupby(T.day.dt.to_period("W")).pnl.sum()
    P(f"  週：正 {(wk > 0).mean() * 100:.0f}%；最差週 {wk.min():+.0f}，最好週 {wk.max():+.0f}")


def sizing(T, P, start=10000.0):
    P("\n  資金模擬（起始 10,000 美元，每筆風險 = 權益 r%，手數 = 風險 / (停損距離 × 100)，最小 0.01 手）")
    T = T.sort_values("exit_time")
    for cap_n in (99, 3):
        for r in (0.25, 0.5, 1.0):
            eq = peak = start; mdd = 0.0; open_ = []; lots = []
            for x in T.itertuples():
                open_ = [e for e in open_ if e > x.entry_time]
                if len(open_) >= cap_n:
                    continue
                open_.append(x.exit_time)
                lot = max(0.01, round(eq * r / 100 / (x.risk * 100), 2)); lots.append(lot)
                eq += lot * 100 * x.pnl; peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
            P(f"    上限 {cap_n if cap_n < 99 else '無':>2} 筆、每筆 {r:4}%：期末 {eq:10,.0f}（{(eq / start - 1) * 100:+6.0f}%）"
              f" 最大回撤 {mdd * 100:5.1f}%  手數中位數 {np.median(lots):.2f}")


if __name__ == "__main__":
    out = open("results_final.txt", "w")
    def P(x=""): print(x); out.write(x + "\n"); out.flush()
    X = simulate(raw_signals())
    IS = X[X.day < SPLIT]
    q = thresholds(IS)
    P(f"固定門檻（1~5 月分位數）：thr_ib = {q[0]:.3f} ATR、thr_tail = {q[1]:.3f} ATR、thr_ext = {q[2]:.2f}")
    F = apply_rules(X, *q)
    keep = ["day", "h", "side", "entry_time", "lvl", "ibh", "ibl", "risk", "exit_time", "exit_px", "exit_reason", "pnl", "R",
            "mae", "mfe", "hold_h", "atr", "ib_atr", "aligned", "tail", "ext_letter", "spread"]
    Fo = F[keep].rename(columns={"lvl": "entry_px", "h": "ib_hour"}).sort_values("entry_time")
    Fo.to_csv("final_trades.csv", index=False, float_format="%.4f")
    report(F, P, "固定門檻 — 全期")
    report(F[F.day < SPLIT], P, "固定門檻 — 1~5 月（挑參數期）")
    report(F[F.day >= SPLIT], P, "固定門檻 — 6~10 月（驗證期）")
    sizing(F, P)
    WF, used = walk_forward(X)
    P("\n滾動前推：每月月初只用過去資料重算門檻")
    P(used.round(3).to_string(index=False))
    report(WF, P, "滾動前推 — 3~10 月（每個月都是樣本外）")
    sizing(WF, P)
    WF[keep].rename(columns={"lvl": "entry_px", "h": "ib_hour"}).sort_values("entry_time").to_csv(
        "final_trades_walkforward.csv", index=False, float_format="%.4f")
    out.close()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for T, lab, col in [(F, "fixed thresholds (Jan-May quantiles)", "#3b6ea8"), (WF, "walk-forward thresholds", "#c9824a")]:
        T = T.sort_values("exit_time"); eq = T.pnl.cumsum()
        ax[0, 0].plot(T.exit_time, eq, label=lab, color=col); ax[0, 1].plot(T.exit_time, eq - eq.cummax(), color=col)
    ax[0, 0].axvline(SPLIT, color="gray", ls="--"); ax[0, 0].legend(); ax[0, 0].set_title("cumulative PnL (USD/oz)"); ax[0, 0].grid(alpha=.3)
    ax[0, 1].set_title("drawdown (USD/oz)"); ax[0, 1].grid(alpha=.3)
    ax[1, 0].hist(F.R.clip(-1.5, 8), bins=40, color="#3b6ea8"); ax[1, 0].set_title("R multiple per trade (fixed)"); ax[1, 0].grid(alpha=.3)
    c = np.where(F.pnl > 0, "#2e8b57", "#c0392b")
    ax[1, 1].scatter(F.mae / F.risk, F.mfe / F.risk, c=c, s=10, alpha=.6)
    ax[1, 1].set_xlabel("MAE / risk"); ax[1, 1].set_ylabel("MFE / risk"); ax[1, 1].set_yscale("symlog")
    ax[1, 1].set_title("MAE vs MFE (green = winner)"); ax[1, 1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("final_system.png", dpi=110)
