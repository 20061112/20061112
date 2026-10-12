"""第二十二部分 B：把發布前後的現象做成可執行規則（避開發布那一分鐘成交），以及版本 A 持倉遇到數據的處理。"""
import numpy as np
import pandas as pd
import part22_events as E          # 重用資料、日曆、事件表（會重寫 results_part22.txt 的前段）

P, C, H, L, O, T, SP, pos = E.P, E.C, E.H, E.L, E.O, E.T, E.SP, E.pos
SLIP = 0.30                        # 發布後 15 分鐘內成交，每邊再加 0.3 美元滑價（M1 的 spread 是最小值，低估新聞時段成本）


def day_end(day):
    return np.searchsorted(T, np.datetime64(day + pd.Timedelta(hours=23, minutes=58)))


def simulate(side, e_i, entry, stop, x_i):
    """M1 逐根：停損優先；回傳出場價。"""
    for j in range(e_i, x_i + 1):
        if (side == 1 and L[j] <= stop) or (side == -1 and H[j] >= stop):
            return stop if j > e_i else min(stop, C[j]) if side == 1 else max(stop, C[j])
    return C[x_i]


def summarize(R, name):
    if not len(R):
        P(f"  {name}: 無交易"); return
    R = pd.DataFrame(R)
    R["era"] = np.where(R.t < pd.Timestamp("2025-01-01"), "23-24", "25-26")
    g = R.groupby("era").R.agg(["mean", "size"])
    gp, gl = R.R[R.R > 0].sum(), -R.R[R.R < 0].sum()
    P(f"  {name:46s} n={len(R):3d} 每筆 {R.R.mean():+.2f}R t={E.tstat(R.R):+.1f} 勝率 {(R.R > 0).mean():.0%} PF {gp / max(gl, 1e-9):.2f} "
      f"每筆 {R.atr_pnl.mean():+.3f}ATR | " + "  ".join(f"{k}: {v['mean']:+.2f}R({int(v['size'])})" for k, v in g.iterrows()))


# ---------- 1. 新聞 IB 突破：等發布後 k 分鐘、M1 收盤在 IB 外才進 ----------
def news_ib_close(X, wait, hz):
    R = []
    for r in X.itertuples():
        i = pos(r.t)
        if i >= len(T) or T[i] != np.datetime64(r.t):
            continue
        hi, lo = H[i - 60:i].max(), L[i - 60:i].min()
        de = day_end(r.day)
        for j in range(i + wait, min(i + 60, de)):
            side = 1 if C[j] > hi else -1 if C[j] < lo else 0
            if side:
                entry = C[j] + side * (SP[j] + SLIP)
                stop = lo if side == 1 else hi
                if side * (entry - stop) <= 0:
                    break
                x_i = de - 1 if hz == "eod" else min(j + hz, de - 1)
                ex = simulate(side, j + 1, entry, stop, x_i) - side * SLIP
                pnl = side * (ex - entry)
                R.append(dict(t=r.t, R=pnl / abs(entry - stop), atr_pnl=pnl / r.atr))
                break
    return R


P("\n" + "=" * 110)
P("第二十二部分 B：可執行版本（發布那一分鐘不成交；進出場各加 0.3 美元滑價 + 當根點差）")
P("\n## B1. 新聞 IB（發布前 60 分鐘高低）— 發布後第 wait 分鐘起，M1 收盤在 IB 外才順勢進場；停損 IB 另一側")
for e, X in (("NFP", E.EV["NFP"]), ("CPI", E.EV["CPI"]), ("NFP+CPI", E.EV["NFP+CPI"]), ("FOMC", E.EV["FOMC"]),
             ("CLAIMS", E.EV["CLAIMS"]), ("對照15:30", E.CTL1530), ("對照週三21:00", E.CTL2100)):
    for wait in (1, 5, 15):
        summarize(news_ib_close(X, wait, "eod"), f"{e} wait={wait} 收盤出場")


# ---------- 2. 第一波 15 分鐘後：反向（NFP）/ 順向（FOMC） ----------
def first_wave(X, mode, k=15, hz=45, big=None):
    R = []
    for r in X.itertuples():
        i = pos(r.t)
        if i >= len(T) or T[i] != np.datetime64(r.t):
            continue
        de = day_end(r.day)
        j = i + k - 1
        if j + 1 >= de:
            continue
        mv = C[j] - C[i - 1]
        if mv == 0 or (big is not None and abs(mv) / r.atr < big):
            continue
        s = np.sign(mv) * (1 if mode == "follow" else -1)
        hi, lo = H[i:j + 1].max(), L[i:j + 1].min()
        if mode == "fade":           # 反向：停損在第一波極值外
            stop = hi if s == -1 else lo
        else:                        # 順向：停損在第一波起點（發布前價格）
            stop = C[i - 1]
        entry = C[j] + s * (SP[j] + SLIP)
        if s * (entry - stop) <= 0:
            continue
        x_i = de - 1 if hz == "eod" else min(j + hz, de - 1)
        ex = simulate(int(s), j + 1, entry, stop, x_i) - s * SLIP
        pnl = s * (ex - entry)
        R.append(dict(t=r.t, R=pnl / abs(entry - stop), atr_pnl=pnl / r.atr))
    return R


P("\n## B2. 第一波（0→15 分）之後")
for e, X in (("NFP", E.EV["NFP"]), ("CPI", E.EV["CPI"]), ("NFP+CPI", E.EV["NFP+CPI"]), ("FOMC", E.EV["FOMC"]),
             ("CLAIMS", E.EV["CLAIMS"]), ("對照15:30", E.CTL1530), ("對照週三21:00", E.CTL2100)):
    summarize(first_wave(X, "fade", hz=45), f"{e} 反向 15→60，停損第一波極值")
    summarize(first_wave(X, "follow", hz="eod"), f"{e} 順向 15→收盤，停損發布前價")
    summarize(first_wave(X, "follow", hz="eod", big=0.25), f"{e} 順向（|第一波|≥0.25ATR）15→收盤")


# ---------- 3. ER30 過濾第一波延續（第 4 節唯一兩期同號的 NFP+CPI 因子） ----------
P("\n## B3. 發布前 30 分鐘 ER 與第一波延續（NFP+CPI）：sign(0→15)×(15→收盤) ÷ATR，依 |ER30| 三分位")
X = E.EV["NFP+CPI"].copy()
X["C"] = np.sign(X.r15) * (X.r_eod - X.r15); X["aer"] = X.er30.abs(); X["era"] = E.era(X)
q = X.aer.quantile([1 / 3, 2 / 3]).values
X["grp"] = np.where(X.aer < q[0], "低", np.where(X.aer < q[1], "中", "高"))
for g in ("低", "中", "高"):
    Y = X[X.grp == g]
    P(f"  |ER30| {g}（界線 {q[0]:.2f}/{q[1]:.2f}）n={len(Y)} 平均 {Y.C.mean():+.3f}({E.tstat(Y.C):+.1f}) 勝率 {(Y.C > 0).mean():.0%} | "
      + "  ".join(f"{k}: {v.mean():+.3f}({len(v)})" for k, v in Y.groupby('era').C))
Z = E.EV["CLAIMS"].copy(); Z["C"] = np.sign(Z.r15) * (Z.r_eod - Z.r15); Z["aer"] = Z.er30.abs()
qq = Z.aer.quantile([1 / 3, 2 / 3]).values
P("  初領失業金同樣分法：" + "  ".join(f"{g} {Z[m].C.mean():+.3f}({E.tstat(Z[m].C):+.1f})" for g, m in
      (("低", Z.aer < qq[0]), ("中", (Z.aer >= qq[0]) & (Z.aer < qq[1])), ("高", Z.aer >= qq[1]))))


# ---------- 4. 版本 A（2026/3~10）持倉遇到數據 ----------
P("\n## B4. 版本 A 的單在發布時還持有：抱著 vs 發布前 1 分鐘先平")
A = pd.read_csv("final_A_trades.csv", encoding="utf-8-sig", parse_dates=["進場時間", "出場時間"])
A["side"] = np.where(A["方向"] == "多", 1, -1)
cal = E.CAL[E.CAL.event != "CLAIMS"]
rows = []
for ev in cal.itertuples():
    held = A[(A["進場時間"] < ev.t) & (A["出場時間"] > ev.t)]
    i = pos(ev.t)
    if i >= len(T):
        continue
    for _, r in held.iterrows():
        p_pre = C[i - 1]
        ent, risk, why = r["進場價"], r["停損距離"], r["出場原因"]
        alt = r.side * (p_pre - ent) - r["點差"]           # 發布前一根收盤平倉（扣點差）
        rows.append(dict(event=ev.event, t=ev.t, side=r.side, pnl=r["損益_美元每盎司"], alt=alt, risk=risk,
                         stop_after=(why == "停損") and (r["出場時間"] - ev.t) <= pd.Timedelta(minutes=60),
                         float_R=r.side * (p_pre - ent) / risk))
H4 = pd.DataFrame(rows)
if len(H4):
    for e, Y in list(H4.groupby("event")) + [("全部", H4)]:
        P(f"  {e:5s} 持有中的單 {len(Y):3d} 筆（{Y.t.nunique()} 次數據）  抱著：每筆 {Y.pnl.mean():+7.2f}  總 {Y.pnl.sum():+8.1f} | "
          f"發布前平倉：每筆 {Y.alt.mean():+7.2f}  總 {Y.alt.sum():+8.1f} | 發布後 60 分內被停損 {Y.stop_after.mean():.0%}")
    for lab, Y in (("發布前已浮盈 ≥ 1R", H4[H4.float_R >= 1]), ("發布前浮盈 0~1R", H4[(H4.float_R >= 0) & (H4.float_R < 1)]),
                   ("發布前浮虧", H4[H4.float_R < 0])):
        P(f"  {lab:16s} {len(Y):3d} 筆  抱著 總 {Y.pnl.sum():+8.1f} | 先平 總 {Y.alt.sum():+8.1f}")
# 版本 A 在 15:30 前後的新進場
A["ev_day"] = A["交易日"].isin(set(cal[cal.event.isin(["NFP", "CPI"])].date.dt.strftime("%Y-%m-%d")))
P(f"  版本 A：NFP/CPI 當天的單 {A.ev_day.sum()} 筆 每筆 {A[A.ev_day]['損益_美元每盎司'].mean():+.2f}"
  f"（PF {A[A.ev_day]['損益_美元每盎司'].clip(lower=0).sum() / -A[A.ev_day]['損益_美元每盎司'].clip(upper=0).sum():.2f}）"
  f"；其他日 {(~A.ev_day).sum()} 筆 每筆 {A[~A.ev_day]['損益_美元每盎司'].mean():+.2f}"
  f"（PF {A[~A.ev_day]['損益_美元每盎司'].clip(lower=0).sum() / -A[~A.ev_day]['損益_美元每盎司'].clip(upper=0).sum():.2f}）")
