"""定投 QLD 當底倉，平靜上升時一樣換 TQQQ。三種標的（QQQ 1x / QLD 2x / TQQQ 3x）。

  BASE  原策略：QQQ ↔ TQQQ
  V1    QLD ↔ TQQQ（原報告測過的「QLD 當底倉」）
  V2    三段：MA200 下 → QQQ；MA200 上但 z ≥ 0.5 → QLD；MA200 上且 z < 0.5 → TQQQ
  V3    同 V2，但 MA200 下且 z < 0.5 → QLD
規則同原策略：往上加槓桿只在每週第一個交易日，往下降槓桿每天檢查；定投 80% 進當前標的、20% 存現金；z > 2 現金補 QQQ。
QLD 2010 年前用合成：2 × 日報酬 − 0.95% − 1 ×（T-bill + 0.45%），先用 2006 起的真實 QLD 校準。
輸出 results_qld.txt。
"""
import numpy as np
import pandas as pd
import engine as E
import indicators as ind
from run_tests import q, OK0, Z, ABOVE

out = []
P = lambda s: (print(s, flush=True), out.append(s))


def add_q2(d):
    rf_y = d.rf * 252
    r2 = 2 * d.r - E.FEE / 252 - (rf_y + E.SPREAD) / 252
    r2.iloc[0] = 0.0
    d["q2_syn"] = (1 + r2).cumprod()
    return d


def levels(d, z, above, kind):
    """回傳每天「訊號上」的目標槓桿 1/2/3（未 shift）。"""
    calm = z < 0.5
    L = pd.Series(1, index=d.index)
    if kind == "BASE":
        L[above & calm] = 3
    elif kind == "V1":
        L[:] = 2; L[above & calm] = 3
    elif kind == "V2":
        L[above] = 2; L[above & calm] = 3
    elif kind == "V3":
        L[above | calm] = 2; L[above & calm] = 3
    return L


def apply_timing(d, L):
    """前一天訊號；往上只在週初、往下每天。"""
    t = L.shift(1).fillna(1).astype(int).values
    ws = d.week_start.values
    out_ = np.ones(len(t), int); cur = int(t[0])
    for i in range(len(t)):
        if t[i] < cur or ws[i]:
            cur = t[i]
        out_[i] = cur
    return out_


def run3(d, lev, start=None, end=None, z=None, cols=("close", "q2_syn", "tq_syn")):
    idx = d.index
    i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    px = {1: d[cols[0]].values, 2: d[cols[1]].values, 3: d[cols[2]].values}
    rf = d.rf.values; ws = d.week_start.values
    zp = (E.atr_z(d) if z is None else z).shift(1).values
    bzp = E.boll_z(d).shift(1).values
    units = {1: 0.0, 2: 0.0, 3: 0.0}; cash = 0.0; inv = 0.0; flows = []; sw = 0
    vals = np.zeros(i1 - i0); invs = np.zeros(i1 - i0); first = True
    for k, i in enumerate(range(i0, i1)):
        cash *= 1 + rf[i]
        tgt = lev[i]
        if not np.isfinite(px[tgt][i]):
            tgt = 1
        if ws[i] or first:
            first = False
            amt = 2.0 if (np.isfinite(bzp[i]) and bzp[i] <= -0.6) else 1.0
            inv += amt; flows.append((idx[i], -amt))
            units[tgt] += 0.8 * amt / px[tgt][i]; cash += 0.2 * amt
            if np.isfinite(zp[i]) and zp[i] > 2 and cash > 0:
                units[1] += cash / px[1][i]; cash = 0.0
        held = {a: units[a] * px[a][i] for a in units if units[a] > 0}
        tot = sum(held.values())
        if tot > 0 and any(a != tgt for a in held):
            moved = tot - held.get(tgt, 0.0)
            tot -= moved * E.COST
            units = {1: 0.0, 2: 0.0, 3: 0.0}; units[tgt] = tot / px[tgt][i]; sw += 1
        vals[k] = sum(units[a] * px[a][i] for a in units if units[a] > 0) + cash
        invs[k] = inv
    flows.append((idx[i1 - 1], vals[-1]))
    return dict(value=vals, invested=invs, flows=flows, switches=sw, years=(idx[i1 - 1] - idx[i0]).days / 365.25)


if __name__ == "__main__":
    add_q2(q)
    # 校準
    real = E._read("qld_us_d.csv").close.reindex(q.index)
    m = real.notna()
    rr = real[m].pct_change().dropna(); rs = q.q2_syn[m].pct_change().dropna()
    yrs = (rr.index[-1] - rr.index[0]).days / 365.25
    cagr = lambda r: (1 + r).prod() ** (1 / yrs) - 1
    P(f"合成 QLD 校準（{rr.index[0].date()} 起）：真實年化 {cagr(rr):.2%}、合成 {cagr(rs):.2%}、差 {cagr(rs) - cagr(rr):+.2%}/年、"
      f"追蹤誤差 {(rr - rs).std() * np.sqrt(252):.2%}")
    q["q2_mix"] = q.q2_syn.where(~m, np.nan)
    # 真實 QLD 接在合成後面（比例銜接）
    first = real[m].index[0]
    q["q2_real"] = np.where(q.index < first, q.q2_syn, real / real[first] * q.q2_syn[first])

    rows = []
    base_res = {}
    for kind in ["BASE", "V1", "V2", "V3"]:
        lev = apply_timing(q, levels(q, Z, ABOVE, kind))
        f = E.stats(run3(q, lev))
        s10 = E.stats(run3(q, lev, start="2010-02-12", cols=("close", "q2_real", "tq_real")))
        roll = []
        for y in range(2000, 2017):
            roll.append(E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01")) | dict(start=y))
        roll = pd.DataFrame(roll)
        if kind == "BASE":
            base_res = roll
        d = roll.irr.values - base_res.irr.values
        avg = pd.Series(lev).value_counts(normalize=True).reindex([1, 2, 3]).fillna(0)
        rows.append(dict(kind=kind, irr99=f["irr"], irr10=s10["irr"], worst=f["worst"], sw=f["sw_per_year"],
                         d10=np.median(d), win10=int((d > 0).sum()), g10min=roll.gain.min(), g10mean=roll.gain.mean(),
                         t1=avg[1], t2=avg[2], t3=avg[3]))
        P(f"{kind:4s} 1999 起 IRR {f['irr']:.3f}｜2010 起（真實 QLD/TQQQ）{s10['irr']:.3f}｜最慘 {f['worst']:.2f}｜"
          f"10 年 IRR 差中位 {np.median(d):+.3f}（{(d > 0).sum()}/17）｜10 年平均 {roll.gain.mean():+.0%}、最差 {roll.gain.min():+.0%}｜"
          f"換倉 {f['sw_per_year']:.1f}/年｜時間占比 1x {avg[1]:.0%} 2x {avg[2]:.0%} 3x {avg[3]:.0%}")
        # 最慘出現在哪一段
        if kind != "BASE":
            pass
    # 逐段 10 年最慘相對本金
    P("\n10 年滾動各段「最慘相對本金」")
    tab = {}
    for kind in ["BASE", "V1", "V2", "V3"]:
        lev = apply_timing(q, levels(q, Z, ABOVE, kind))
        tab[kind] = [E.stats(run3(q, lev, f"{y}-01-01", f"{y + 10}-01-01"))["worst"] for y in range(2000, 2017)]
    T = pd.DataFrame(tab, index=range(2000, 2017)).T
    P(T.to_string(float_format=lambda x: f"{x:.0%}"))

    # 蒙地卡羅
    import run_ideas_mc as MC
    res = []
    for k in range(300):
        d = MC.path(); add_q2(d); z = E.atr_z(d); above = d.close > d.close.rolling(200).mean()
        s0 = d.index[MC.WARM]; row = {}
        for kind in ["BASE", "V1", "V2", "V3"]:
            st = E.stats(run3(d, apply_timing(d, levels(d, z, above, kind)), start=s0, z=z))
            row[kind] = st["irr"]; row[kind + "|w"] = st["worst"]
        res.append(row)
    M = pd.DataFrame(res)
    P("\n區塊拼貼蒙地卡羅（300 條 10 年路徑）")
    P(f"  BASE IRR 中位 {M.BASE.median():.3f}、5% {M.BASE.quantile(.05):.3f}；最慘中位 {M['BASE|w'].median():.2f}、5% {M['BASE|w'].quantile(.05):.2f}")
    for kind in ["V1", "V2", "V3"]:
        dd = M[kind] - M.BASE; dw = M[kind + "|w"] - M["BASE|w"]
        P(f"  {kind} − BASE：IRR 差 中位 {dd.median():+.4f} 勝率 {(dd > 0).mean():.0%} 5% {dd.quantile(.05):+.4f}｜"
          f"最慘 差 中位 {dw.median():+.3f} 5% {dw.quantile(.05):+.3f}（{kind} 最慘 5% {M[kind + '|w'].quantile(.05):.2f}）")
    open("results_qld.txt", "w").write("\n".join(out) + "\n")
