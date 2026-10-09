"""第二十三部分 A：第二十二部分「回復力切換」策略的逐筆交易明細。

規則（broker 時間，M5，資料 data/XAUUSD_M1_2026.csv）：
  均值 EMA20；D = (收盤 − EMA20) / ATR14；RFI-U = 第二十二部分 I 段的 8 個量測（只用 1~5 月定方向與分位）。
  訊號：|D| 第一次 ≥ 2.0（回到 1.0 以內才重新武裝），在 K 棒收盤判斷。
    RFI ≥ IS 2/3 分位 → 反向（回歸）；RFI ≤ IS 1/3 分位 → 順向（動能）；中間不做。
  進場：下一根 M5 開盤；停損 = 進場 ∓ 2 × ATR14；出場：停損，或持有 12 根（60 分鐘）後收盤。
  同時只持有一筆；每筆扣進場當下點差。金額 = 美元/盎司（= 0.01 手的美元）。
輸出：part23_trades.csv（逐筆）、part23_stats.txt（統計）、part23_equity.png。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from part22_lib import load_tf, means, atr, events, CUT
from part22_beta import frame
from part22_final import universal_selection, rfi_u

D0, HOLD, STOP = 2.0, 12, 2.0
SESS = lambda h: "亞洲" if h < 10 else "倫敦" if h < 15 else "紐約" if h < 20 else "尾盤"


def run(d, A, D, R, lo, hi, d0=D0, hold=HOLD, stop=STOP, rule=None):
    """rule(t) → +1 / −1 / 0（交易方向）；預設為 RFI 切換。"""
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    spr = d.spread.to_numpy() / 100; seg = d.seg.to_numpy(); a = A.to_numpy()
    Dv, Rv = D.to_numpy(), R.to_numpy()
    ev = events(D, d.seg, d0=d0, rearm=d0 / 2)
    rows, busy = [], -1
    for t in ev:
        if t < 600 or t <= busy or t + 2 >= len(c) or np.isnan(Rv[t]):
            continue
        if rule is None:
            if Rv[t] >= hi:
                leg, s = "回歸", -np.sign(Dv[t])
            elif Rv[t] <= lo:
                leg, s = "動能", np.sign(Dv[t])
            else:
                continue
        else:
            s, leg = rule(t)
            if s == 0:
                continue
        e = t + 1; ep = o[e]; st = ep - s * stop * a[t]
        mae = mfe = 0.0; xp = xi = None; why = "時間"
        for j in range(e, min(e + hold, len(c))):
            if seg[j] != seg[t]:
                xp, xi, why = c[j - 1], j - 1, "斷線"; break
            adv = (ep - l[j]) if s > 0 else (h[j] - ep)
            fav = (h[j] - ep) if s > 0 else (ep - l[j])
            if (l[j] <= st) if s > 0 else (h[j] >= st):
                xp = (min(st, o[j]) if s > 0 else max(st, o[j])) if j > e else st
                xi, why = j, "停損"; mae = max(mae, s * (ep - xp)); break
            mae, mfe = max(mae, adv), max(mfe, fav)
        if xp is None:
            xi = min(e + hold - 1, len(c) - 1); xp = c[xi]
        pnl = s * (xp - ep) - spr[e]
        rows.append(dict(訊號時間=d.index[t], 進場時間=d.index[e], 出場時間=d.index[xi] + pd.Timedelta(minutes=5),
                         類型=leg, 方向="多" if s > 0 else "空", 進場價=round(ep, 2), 停損價=round(st, 2), 出場價=round(xp, 2),
                         出場原因=why, 損益_美元=round(pnl, 2), 損益_ATR=round(pnl / a[t], 3), 損益_R=round(pnl / (stop * a[t]), 3),
                         最大浮虧_美元=round(mae, 2), 最大浮盈_美元=round(mfe, 2), 持倉_分鐘=(xi - e + 1) * 5,
                         ATR14=round(a[t], 2), D=round(Dv[t], 2), RFI=round(Rv[t], 3), 點差=round(spr[e], 2),
                         時段=SESS(d.index[t].hour), 期間="樣本內" if d.index[t] < CUT else "樣本外"))
        busy = xi
    return pd.DataFrame(rows)


def report(T, title):
    L = [title, "=" * 100]
    pf = lambda p: p[p > 0].sum() / -p[p <= 0].sum() if (p <= 0).any() else np.inf

    def line(s, lab):
        p = s["損益_美元"]
        if len(s) == 0:
            return f"  {lab:14s}    0 筆"
        w, ls = p[p > 0], p[p <= 0]
        eq = p.cumsum(); dd = (eq - eq.cummax()).min()
        return (f"  {lab:14s} {len(s):4d} 筆  勝率 {len(w) / len(s):5.1%}  均賺 {w.mean():+6.2f}  均賠 {ls.mean():+6.2f}  "
                f"賺賠比 {w.mean() / -ls.mean():4.2f}  每筆 {p.mean():+5.2f} 美元 / {s['損益_ATR'].mean():+.3f} ATR  "
                f"PF {pf(p):4.2f}  總 {p.sum():+8.1f}  最大回撤 {dd:+7.1f}")
    L.append(line(T, "全部"))
    for k in ("樣本內", "樣本外"):
        L.append(line(T[T["期間"] == k], k))
    L.append("\n依類型")
    for k in ("回歸", "動能"):
        for per in ("樣本內", "樣本外"):
            L.append(line(T[(T["類型"] == k) & (T["期間"] == per)], f"{k} {per}"))
    L.append("\n依時段（全期）")
    for k in ("亞洲", "倫敦", "紐約", "尾盤"):
        L.append(line(T[T["時段"] == k], k))
    L.append("\n依出場原因")
    for k, g in T.groupby("出場原因"):
        L.append(line(g, k))
    L.append("\n依月份")
    for k, g in T.groupby(T["進場時間"].dt.to_period("M")):
        L.append(line(g, str(k)))
    p = T["損益_美元"].to_numpy()
    streak = cur = 0
    for x in p:
        cur = cur + 1 if x <= 0 else 0; streak = max(streak, cur)
    days = T.groupby(T["進場時間"].dt.normalize())["損益_美元"].sum()
    L.append(f"\n交易日數 {days.size}，平均每個有單的日子 {len(T) / days.size:.2f} 筆；日勝率 {np.mean(days > 0):.0%}；"
             f"日 Sharpe（年化 252） {days.mean() / days.std() * np.sqrt(252):.2f}")
    L.append(f"最長連虧 {streak} 筆；平均持倉 {T['持倉_分鐘'].mean():.0f} 分鐘；停損比例 {np.mean(T['出場原因'] == '停損'):.0%}")
    L.append(f"平均 ATR14（M5）{T['ATR14'].mean():.2f} 美元；平均點差 {T['點差'].mean():.2f} 美元（約 {np.mean(T['點差'] / T['ATR14']):.3f} ATR）")
    L.append(f"每筆多扣 0.3 美元滑價：每筆 {(T['損益_美元'] - 0.3).mean():+.2f}，PF {pf(T['損益_美元'] - 0.3):.2f}")
    return "\n".join(L)


if __name__ == "__main__":
    keep = universal_selection(pd.read_csv("part22_beta.csv"))
    d = load_tf("M5"); A = atr(d, 14); M = means(d)
    X = frame(d, A, M, "ema20"); X["rfi"] = rfi_u(X, keep)
    lo, hi = np.nanquantile(X.loc[X.per != "OOS", "rfi"], [1 / 3, 2 / 3])
    T = run(d, A, X.D.reindex(d.index), X.rfi.reindex(d.index), lo, hi)
    T.to_csv("part23_trades.csv", index=False, encoding="utf-8-sig")
    txt = report(T, f"第二十二部分回復力切換策略（EMA20、|D|≥{D0}、持有 {HOLD} 根、停損 {STOP} ATR；RFI 分位 {lo:+.3f} / {hi:+.3f}）")
    open("part23_stats.txt", "w").write(txt + "\n"); print(txt)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for k, col in (("回歸", "tab:green"), ("動能", "tab:red")):
        s = T[T["類型"] == k]
        ax.plot(s["進場時間"], s["損益_美元"].cumsum(), color=col, label={"回歸": "mean-reversion leg", "動能": "momentum leg"}[k])
    ax.plot(T["進場時間"], T["損益_美元"].cumsum(), color="tab:blue", lw=2, label="total")
    ax.axvline(CUT, color="k", lw=.8); ax.axhline(0, color="gray", lw=.5)
    ax.set_ylabel("USD per oz (0.01 lot)"); ax.set_title("Part 22 restoring-force switch: cumulative P&L (left of line = in-sample)")
    ax.legend(); fig.tight_layout(); fig.savefig("part23_equity.png", dpi=110)
