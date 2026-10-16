"""期貨槓桿，「原型（QQQ / SPY / DIA）絕對不賣」。

保證金只能來自現金（定投中存成現金的部分，放 T-bill）。現金不夠時三種處理：
  DEBT     允許現金變負（券商融資，利率 T-bill + 1%）；有負債時，定投全部先還債
  DELEVER  現金低於維持保證金（期貨名目 7%）就減少期貨口數；單日虧損超過現金仍可能短暫負債，定投先還債
  兩者都搭配不同的現金比例（定投中存現金的比例 20% / 30% / 40% / 50%）
原型永遠不賣：恐慌補倉（ATR z > 2）只用超過保證金需求的現金買原型。
期貨每 1 元名目每日損益 = 價格報酬 + 股息率 / 252 − 無風險利率；股息率 QQQ 0.7%、SPY 1.4%、DIA 1.8%；每季轉倉 0.02%、每次調整 0.02%。
訊號與原策略相同（各自 MA200 & ATR z < 0.5，每週進場、每天出場），開啟時目標曝險 = 3 倍淨值。
參考：各指數「賣原型換 3 倍 ETF」的原策略（違反不賣的限制，只當對照）。
輸出 results_futures2.txt、results_futures2.csv。
"""
import numpy as np
import pandas as pd
import engine as E
import lev_lib as L

out = []
P = lambda s: (print(s, flush=True), out.append(s))
ROLL = 0.0002; TRADE = 0.0002; NEG = 0.01; MAINT = 0.07
DIV = {"qqq": 0.007, "spy": 0.014, "dia": 0.018}


def run_nosell(d, w, div, mode="DELEVER", cash_frac=0.2, lev=3.0, start=None, end=None):
    idx = d.index
    i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    px = d.close.values; rf = d.rf.values; ws = d.week_start.values
    z = E.atr_z(d).shift(1).values; bz = E.boll_z(d).shift(1).values
    qs = d.index.to_period("Q"); qs = np.r_[True, qs[1:] != qs[:-1]]
    nq = cash = fs = inv = 0.0; flows = []; first = True; on_prev = False
    vals = np.zeros(i1 - i0); invs = np.zeros(i1 - i0)
    lev_on = []; target_days = 0; capped_days = 0; min_cash = 0.0; debt_days = 0; max_ltv = 0.0; max_debt_inv = 0.0; longest = run_len = 0
    for k, i in enumerate(range(i0, i1)):
        cash *= 1 + (rf[i] if cash >= 0 else rf[i] + NEG / 252)
        if fs and i > i0:
            cash += fs * px[i - 1] * ((px[i] / px[i - 1] - 1) + div / 252 - rf[i])
            if qs[i]:
                cash -= fs * px[i] * ROLL
        if ws[i] or first:
            first = False
            amt = 2.0 if (np.isfinite(bz[i]) and bz[i] <= -0.6) else 1.0
            inv += amt; flows.append((idx[i], -amt))
            if cash < 0 and mode != "NOREPAY":
                rep = min(amt, -cash); cash += rep; amt_left = amt - rep
            else:
                amt_left = amt
            nq += (1 - cash_frac) * amt_left / px[i]; cash += cash_frac * amt_left
            if np.isfinite(z[i]) and z[i] > 2:
                spare = cash - MAINT * fs * px[i]
                if spare > 0:
                    nq += spare / px[i]; cash -= spare
        nav = nq * px[i] + cash
        on = w[i] > 0
        target = max((lev * nav - nq * px[i]) / px[i], 0.0) if on else 0.0
        reset = (on != on_prev) or (on and ws[i])
        new_fs = target if reset else fs
        if not on:
            new_fs = 0.0
        if mode == "DELEVER":
            cap = max(cash, 0.0) / (MAINT * px[i])
            if new_fs > cap:
                new_fs = cap
        if abs(new_fs - fs) > 1e-12:
            cash -= abs(new_fs - fs) * px[i] * TRADE
            fs = new_fs
        on_prev = on
        nav = nq * px[i] + cash
        if on:
            target_days += 1
            lev_on.append((nq + fs) * px[i] / nav if nav > 0 else np.nan)
            if fs < target - 1e-9:
                capped_days += 1
        min_cash = min(min_cash, cash / nav if nav > 0 else 0)
        debt_days += cash < 0
        if cash < 0 and nq > 0:
            max_ltv = max(max_ltv, -cash / (nq * px[i]))
            max_debt_inv = max(max_debt_inv, -cash / max(inv, 1e-9))
        run_len = run_len + 1 if cash < 0 else 0
        longest = max(longest, run_len)
        vals[k] = nav; invs[k] = inv
    flows.append((idx[i1 - 1], vals[-1]))
    lv = np.array(lev_on, float)
    n = i1 - i0
    return dict(value=vals, invested=invs, flows=flows, switches=0, years=(idx[i1 - 1] - idx[i0]).days / 365.25,
                lev_med=np.nanmedian(lv) if len(lv) else np.nan, lev_p5=np.nanpercentile(lv, 5) if len(lv) else np.nan,
                capped=capped_days / max(target_days, 1), min_cash=min_cash, debt_frac=debt_days / n, max_ltv=max_ltv, max_debt_inv=max_debt_inv, longest_debt_years=longest / 252)


def show(label, runner, starts, extra=True):
    res = runner(None, None); f = E.stats(res)
    s10 = E.stats(runner("2010-03-12", None))
    roll = pd.DataFrame([E.stats(runner(f"{y}-01-01", f"{y + 10}-01-01")) for y in starts])
    msg = (f"{label:44s} IRR {f['irr']:6.1%}｜2010 起 {s10['irr']:6.1%}｜10 年滾動中位 {roll.irr.median():6.1%} 最差 {roll.irr.min():6.1%}｜最慘 {f['worst']:4.0%}")
    row = dict(label=label, irr=f["irr"], irr10=s10["irr"], roll_med=roll.irr.median(), roll_min=roll.irr.min(), worst=f["worst"])
    if extra and "lev_med" in res:
        msg += (f"｜訊號開啟時槓桿 中位 {res['lev_med']:.2f}（5% {res['lev_p5']:.2f}）"
                f"｜現金最低 {res['min_cash']:.0%} 淨值、負債天數 {res['debt_frac']:.0%}、最高貸款成數 {res['max_ltv']:.0%}"
                f"、最長連續負債 {res['longest_debt_years']:.1f} 年")
        row |= {k: res[k] for k in ("lev_med", "lev_p5", "min_cash", "debt_frac", "max_ltv")}
    P(msg); return row


rows = []
frames = {}
q = L.build("qqq"); frames["qqq"] = (q, list(range(2000, 2017)))
raw = {t: L.build(t) for t in ["spy", "dia"]}
common = raw["spy"].index.intersection(raw["dia"].index).intersection(q.index)
for t in raw:
    frames[t] = (raw[t].loc[common].copy(), list(range(2006, 2017)))
frames["spy"][0]["real3"] = L.real_series(frames["spy"][0], "upro", 3)
frames["dia"][0]["real3"] = L.real_series(frames["dia"][0], "udow", 3)
q["real3"] = q.x3

for t, (d, starts) in frames.items():
    P(f"\n=== {t.upper()}（{d.index[0].date()} ~ {d.index[-1].date()}，股息率 {DIV[t]:.1%}）")
    _, _, ok = L.signals(d); lev = L.codes(d, ok); w = (lev == 3).astype(float)
    rows.append(show(f"[對照，會賣原型] {t.upper()} ↔ 3 倍 ETF", L.strat(d, lev, ("base", "bill", "real3")), starts, extra=False) | dict(idx=t))
    rows.append(show(f"[對照] 純定投 {t.upper()}", lambda a, b, d=d: L.dca_only(d, a, b), starts, extra=False) | dict(idx=t))
    for mode in ["DEBT", "NOREPAY", "DELEVER"]:
        for cf in [0.2, 0.3, 0.4, 0.5]:
            rows.append(show(f"期貨不賣原型 {mode:7s} 現金 {cf:.0%}", lambda a, b, d=d, w=w, m=mode, c=cf: run_nosell(d, w, DIV[t], m, c, start=a, end=b), starts)
                        | dict(idx=t, mode=mode, cash_frac=cf))

pd.DataFrame(rows).to_csv("results_futures2.csv", index=False, float_format="%.4f")
open("results_futures2.txt", "w").write("\n".join(out) + "\n")
