"""第二十三部分：之前所有策略在 NAS100 的凍結參數測試（2025/10/1 ~ 2026/10/16）。

規則、時段（broker 時間 = 紐約 + 7）、參數一律沿用原檔案，不調整。需要分位數門檻的（第十五~十七部分），
沿用原本「只用過去資料」的做法在 NAS100 上重算；固定數字的門檻（0.13 ATR 等）直接照抄。
NAS100 價格 ÷ 5（tpo.INSTR），TPO 格子相對 ATR 的解析度與黃金相同；R / ATR / PF 不受影響。
M15、M5 由 M1 合成（backtest.resample）。前後兩半以 CUT 切開。

會在 import 時直接執行並覆寫結果檔的舊腳本（run_part10、part10_freq、part11_levels、part12_final、
hypotheses/feature_stability、rules5/select5），這裡只複製它們的規則函式，不 import。
輸出：results_part23.txt
"""
import os
os.environ["SYMBOL"] = "NAS100"
import numpy as np
import pandas as pd
from tpo import load_m1, build_days, TICK, INSTR
from backtest import resample

K = INSTR["NAS100"][1]
CUT = pd.Timestamp("2026-04-01")
M1 = load_m1()
BARS = M1[["open", "high", "low", "close", "tickvol", "spread"]]
M15, M5 = resample(BARS, "15min"), resample(BARS, "5min")
D, PROF = build_days(M1)
DAYS = list(D.index)
G = {k: g for k, g in M1.groupby("day") if k in D.index}
NDAYS = lambda a, b: sum(1 for d in DAYS if a <= d <= b)

out = open("results_part23.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def tst(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def pf(x):
    x = pd.Series(x).dropna()
    return x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-12)


def rline(lab, r, when):
    """逐筆 R（每筆風險固定時的報酬）。"""
    r, when = pd.Series(np.asarray(r, float)), pd.Series(pd.to_datetime(np.asarray(when)))
    if len(r) < 5:
        return f"  {lab:40s} {len(r)} 筆"
    h = (when < CUT).values
    nd = NDAYS(when.min().normalize(), when.max().normalize())
    m = r.groupby(when.dt.to_period("M").values).sum()
    eq = r.cumsum()
    return (f"  {lab:40s} {len(r):5d}筆 {len(r) / nd:4.1f}/日 每筆 {r.mean():+.3f}R t={tst(r):+5.2f} 勝率 {(r > 0).mean() * 100:3.0f}% "
            f"PF {pf(r):.2f} 總 {r.sum():+7.1f}R 回撤 {(eq.cummax() - eq).max():5.1f}R 正月 {(m > 0).sum()}/{len(m)} "
            f"| 前半 {r[h].mean():+.3f} 後半 {r[~h].mean():+.3f}")


def aline(lab, ret_atr, pnl, when):
    """IB 類：每筆報酬以 ATR10 計（不受停損大小影響），PF 用實際點數。"""
    a, p = pd.Series(np.asarray(ret_atr, float)), pd.Series(np.asarray(pnl, float))
    when = pd.Series(pd.to_datetime(np.asarray(when)))
    if len(a) < 5:
        return f"  {lab:40s} {len(a)} 筆"
    h = (when < CUT).values
    nd = NDAYS(when.min().normalize(), when.max().normalize())
    eq = (p * K).cumsum()
    m = p.groupby(when.dt.to_period("M").values).sum()
    return (f"  {lab:40s} {len(a):5d}筆 {len(a) / nd:4.1f}/日 每筆 {a.mean():+.3f}ATR t={tst(a):+5.2f} 勝率 {(p > 0).mean() * 100:3.0f}% "
            f"PF {pf(p):.2f} 總 {p.sum() * K:+7.0f}點 回撤 {(eq - eq.cummax()).min():6.0f}點 正月 {(m > 0).sum()}/{len(m)} "
            f"| 前半 PF {pf(p[h]):.2f} {a[h].mean():+.3f} 後半 PF {pf(p[~h]):.2f} {a[~h].mean():+.3f}")


def dline(lab, day_ret):
    """以「日」為單位的報酬（ATR），t 用日報酬。"""
    x = day_ret.dropna()
    h = x.index < CUT
    cum = x.cumsum()
    return (f"  {lab:40s} {len(x):4d}天 日均 {x.mean():+.3f}ATR t={tst(x):+5.2f} 正報酬天 {(x > 0).mean() * 100:3.0f}% "
            f"累積 {cum.iloc[-1]:+6.1f}ATR 回撤 {(cum - cum.cummax()).min():5.1f}ATR | 前半 {x[h].mean():+.3f} 後半 {x[~h].mean():+.3f}")


P(f"第二十三部分：之前所有策略在 NAS100 的凍結參數測試（{DAYS[0].date()} ~ {DAYS[-1].date()}，{len(DAYS)} 個交易日）")
P(f"前半 < {CUT.date()} ≤ 後半；ATR10 中位數 {D.atr10.median() * K:.0f} 點；點差中位數 {M1.spread.median() * 0.01 * K:.2f} 點")

# =====================================================================================================
P("\n" + "=" * 130 + "\n一、TPO + IB")

# ---- 第十部分：LDN / NY / ASIA IB 首次突破（run_part10.ib_rows 的規則：IB 邊進場、另一側停損、收盤出，未扣成本）----
SESS = {"ASIA": 0, "LDN": 18, "NY": 29}


def ib_events(sp):
    rows = []
    for i, day in enumerate(DAYS):
        if i == 0:
            continue
        g = G[day]
        ib, after = g[(g.period >= sp) & (g.period < sp + 2)], g[g.period >= sp + 2]
        if len(ib) < 50 or len(after) < 60:
            continue
        ibh, ibl = ib.high.max(), ib.low.min(); rng = ibh - ibl
        if rng <= 0:
            continue
        px, atr = ib.close.iloc[-1], D.atr10.iloc[i]
        ppr, _, _, pcz = PROF[DAYS[i - 1]]
        k = int(np.floor((px - ppr[0]) / TICK))
        tree_z = pcz[k] if 0 <= k < len(pcz) else np.nan
        up_t = after.index[after.high > ibh]; dn_t = after.index[after.low < ibl]
        first = 1 if len(up_t) and (not len(dn_t) or up_t[0] < dn_t[0]) else -1 if len(dn_t) else 0
        ret = np.nan
        if first:
            t0 = up_t[0] if first == 1 else dn_t[0]
            lvl, stop = (ibh, ibl) if first == 1 else (ibl, ibh)
            rest = after[after.index >= t0]
            hit = (rest.low <= stop) if first == 1 else (rest.high >= stop)
            ret = first * ((stop if hit.any() else after.close.iloc[-1]) - lvl) / atr
        rows.append(dict(day=day, rng=rng, tree_z=tree_z, ret=ret))
    E = pd.DataFrame(rows).set_index("day")
    lr = np.log(E.rng)
    E["ib_z"] = (lr - lr.rolling(20).mean().shift(1)) / lr.rolling(20).std().shift(1)
    return E


P("\n[第十部分] IB 首次突破 → 收盤（每天一筆，報酬以 ATR 計）")
P("  黃金：LDN 大 IB(z>0.5) +0.17ATR t2.3、排除前日 HVN +0.20ATR；NY 大 IB −0.09ATR（偏失敗）")
for s, sp in SESS.items():
    E = ib_events(sp)
    P(dline(f"{s} 全部", E.ret))
    P(dline(f"{s} 大 IB（z > 0.5）", E.ret[E.ib_z > 0.5]))
    if s == "LDN":
        P(dline("LDN 大 IB + 不在前日 HVN（tree_z ≤ 0.5）", E.ret[(E.ib_z > 0.5) & ~(E.tree_z > 0.5)]))


# ---- 第十部分補充：整點 IB 突破組合（part10_freq.trades / part10_portfolio 的 A~D）----
def freq_trades(h, win=3, rev=False):
    outr = []
    for day in DAYS:
        g = G[day]; atr = D.atr10.get(day); t = g.index
        ib = g[t.hour == h]
        if len(ib) < 50 or not np.isfinite(atr):
            continue
        ibh, ibl = ib.high.max(), ib.low.min(); rng = ibh - ibl
        aft = g[t >= ib.index[-1] + pd.Timedelta(minutes=1)]
        if len(aft) < 30:
            continue
        H, L, C, S = aft.high.values, aft.low.values, aft.close.values, aft.spread.values * 0.01
        tm = aft.index; wend = ib.index[-1] + pd.Timedelta(hours=win); inwin = tm <= wend
        up = np.where((H > ibh) & inwin)[0]; dn = np.where((L < ibl) & inwin)[0]
        if not len(up) and not len(dn):
            outr.append(dict(day=day, rng=rng, ret=np.nan, n=0)); continue
        legs = [(1, up[0])] if len(up) and (not len(dn) or up[0] < dn[0]) else [(-1, dn[0])]
        res, k = [], 0
        while k < len(legs):
            side, i0 = legs[k]
            lvl, stop = (ibh, ibl) if side == 1 else (ibl, ibh)
            hit = np.where((L[i0:] <= stop) if side == 1 else (H[i0:] >= stop))[0]
            px, j = (stop, i0 + hit[0]) if len(hit) else (C[-1], None)
            res.append((side * (px - lvl) - S[i0]) / atr)
            if rev and j is not None and k == 0 and tm[j] <= wend:
                legs.append((-side, j))
            k += 1
        outr.append(dict(day=day, rng=rng, ret=sum(res), n=len(res)))
    E = pd.DataFrame(outr).set_index("day")
    lr = np.log(E.rng)
    E["z"] = (lr - lr.rolling(20).mean().shift(1)) / lr.rolling(20).std().shift(1)
    return E


P("\n[第十部分補充] 整點 IB 突破組合（日報酬 ATR；黃金 A +0.10/筆 t2.7、B t3.1、C t3.3、D +0.04/筆 t2.4）")
for name, legs in {"A 03h（反手）": [(3, 3, True, None)], "B A + 10h z>0（反手）": [(3, 3, True, None), (10, 3, True, 0)],
                   "C B + 05h": [(3, 3, True, None), (10, 3, True, 0), (5, 3, False, None)],
                   "D 01~10h 每個整點（無反手）": [(h, 3, False, None) for h in range(1, 11)]}.items():
    rets = []
    for h, win, rev, zmin in legs:
        E = freq_trades(h, win, rev)
        E = E[E.z.notna() & ((E.z > zmin) if zmin is not None else True)]
        rets.append(E.ret.rename(h))
    R = pd.concat(rets, axis=1).fillna(0)
    P(dline(name, R.sum(axis=1)))

# ---- 第十一部分：碰前日 POC / VAH / VAL 做反轉（part11_levels.event，30 分 TPO）----
ATR11 = (D.high - D.low).rolling(10).mean().shift(1)


def lvl_event(g, L, atr, stop_k=0.1, tgt_k=0.2):
    H, Lo, C, S = g.high.values, g.low.values, g.close.values, g.spread.values * 0.01
    hit = np.where((Lo <= L) & (H >= L) & (g.index.hour < 22))[0]; hit = hit[hit > 0]
    if not len(hit):
        return None
    i = hit[0]; side = np.sign(C[i - 1] - L)
    if side == 0:
        return None
    stop, tgt = L - side * stop_k * atr, L + side * tgt_k * atr
    for j in range(i, min(i + 240, len(C))):
        if (Lo[j] <= stop) if side > 0 else (H[j] >= stop):
            res = -stop_k; break
        if j > i and ((H[j] >= tgt) if side > 0 else (Lo[j] <= tgt)):
            res = tgt_k; break
    else:
        res = side * (C[min(i + 239, len(C) - 1)] - L) / atr
    return res - S[i] / atr


P("\n[第十一部分] 碰到前日 TPO 價位做反轉（停損 0.1 ATR / 停利 0.2 ATR；黃金 −0.011 ATR、勝率 25%，假價位反而較好）")
for off in (0, 0.25, 0.5):
    rows = []
    for a, b in zip(DAYS[:-1], DAYS[1:]):
        atr = ATR11[b]
        if not np.isfinite(atr):
            continue
        for name in ("poc", "vah", "val"):
            for sg in ([1] if off == 0 else [1, -1]):
                r = lvl_event(G[b], D.loc[a, name] + sg * off * atr, atr)
                if r is not None:
                    rows.append(dict(day=b, r=r))
    E = pd.DataFrame(rows)
    P(f"  {'真實價位' if off == 0 else f'對照 ±{off} ATR':40s} {len(E):5d}筆 勝率 {(E.r > 0).mean() * 100:3.0f}%（兩平 33%） "
      f"每筆 {E.r.mean():+.3f}ATR t={tst(E.r):+5.2f} | 前半 {E.r[E.day < CUT].mean():+.3f} 後半 {E.r[E.day >= CUT].mean():+.3f}")

# ---- 第十二部分：單一淨部位 A / B（part12_strategy.backtest + part12_final.FINAL）----
import part12_strategy as p12
P("\n[第十二部分] 單一淨部位（黃金 6~10 月 PF：A 1.49、B 1.45）")
H210 = tuple(range(2, 11))
for name, cfg in {"A 不反手、IB60、停損中點、2h 窗": dict(hours=H210, ib_min=60, win=2, stop="mid", be=None, reverse=False),
                  "B 反手、IB30、停損另一側、1×IB 保本、5h": dict(hours=H210, ib_min=30, win=5, stop="opp", be=1.0, reverse=True)}.items():
    T = p12.backtest(**cfg)
    atr = D.atr10.reindex(T.day).values
    P(aline(name, T.pnl / atr, T.pnl, T.t_in))

# ---- 第十三 / 十四部分：疊單版與規則 ②（part14_deep）----
from part14_deep import signals, exits, rule2
P("\n[第十三 / 十四部分] 疊單 IB 突破（每個 IB 一筆、觸價進場、另一側停損、收盤出；黃金 OOS PF：原始 1.36、②1.50、②+ 1.66）")
B10 = exits(signals(hours=range(1, 11)))
B = exits(signals())
for lab, X in [("原始 01~10 點", B10), ("a+b：02~10 點、IB/ATR ≥ 0.13", B[B.ib_atr >= 0.13]),
               ("規則 ②：不做順前日的小 IB", rule2(B)), ("規則 ②+：再加不做星期一", rule2(B)[rule2(B).day.dt.dayofweek != 0])]:
    P(aline(lab, X.pnl / X.atr, X.pnl, X.t_in))

# ---- 第十五 / 十六部分：最佳系統（final_system：整點 IB + M5 收盤 + tail + ext_letter）----
import final_system as fs
P("\n[第十五 / 十六部分] 最佳系統（整點 IB、M5 收盤確認、tail 80%、ext_letter 80%；黃金滾動前推 PF 2.15、+0.55R）")
XF = fs.simulate(fs.raw_signals())
WF, _ = fs.walk_forward(XF)
P(rline("滾動前推門檻", WF.R, WF.t_in) + f" | PF(點) {pf(WF.pnl):.2f}")
GF = fs.apply_rules(XF, 0.12, 0.03, 0.67)
GF = GF[GF.day >= WF.day.min()]
P(rline("黃金門檻凍結（0.12 / 0.03 / 0.67）", GF.R, GF.t_in) + f" | PF(點) {pf(GF.pnl):.2f}")

# ---- 第十七 / 二十部分：平衡版 B、版本 A + 前一天效率篩選 ----
from part17_balance import gen, wf
P("\n[第十七 / 二十部分]（黃金：版本 A PF 1.92；B PF 1.76；A + 前日效率 PF 2.15）")
A = wf(gen("close5", 0.05, (2, 10), 30, False), 0.9, 0.9)
P(rline("版本 A（參考，同第二十二部分）", A.R, A.t_in) + f" | PF(點) {pf(A.pnl):.2f}")
Bv = wf(gen("close5", 0.05, (1, 12), 30, False), 0.9, 0.9)
P(rline("平衡版 B（01~12 點）", Bv.R, Bv.t_in) + f" | PF(點) {pf(Bv.pnl):.2f}")
eff = ((D.close - D.open).abs() / (D.high - D.low))
A = A.assign(prev_eff=eff.shift(1).reindex(A.day).values, ym=A.day.dt.to_period("M"))
parts = [g[g.prev_eff <= eff[eff.index < m.start_time].quantile(0.75)] for m, g in A.groupby("ym")]
A20 = pd.concat(parts)
P(rline("版本 A + 前一天效率最高 25% 不做", A20.R, A20.t_in) + f" | PF(點) {pf(A20.pnl):.2f}")

# =====================================================================================================
P("\n" + "=" * 130 + "\n二、指標背離")

# ---- 第三部分：M15 反轉型態 + 常見假說（hypotheses.HYP，非重疊）----
from m15_features import candidate_table as m15_table
T3 = m15_table(M15)


def nonoverlap(X):
    X = X.sort_values("entry_time"); keep, busy = [], pd.Timestamp.min
    for r in X.itertuples():
        if r.entry_time > busy:
            keep.append(r.Index); busy = r.exit_time
    return X.loc[keep]


P("\n[第三部分] M15 晨星/夜星/吞噬 + 假說（2R 停利、非重疊；黃金 H4 RSI 背離 −0.07R、H1 +0.29R）")
for name, fn in {"基準：全部候選": lambda X: X.index == X.index,
                 "H1 衝刺竭盡：外軌外收盤 + 加速": lambda X: (X.close_outside == 1) & (X.roc_decel > 0),
                 "H2 擠壓後擴張": lambda X: (X.squeeze_recent == 1) & (X.bbw_chg3 > 0),
                 "H2b 帶寬已在高檔": lambda X: X.bbw_pct > 0.8,
                 "H3 掃前日高/低點後收回": lambda X: X.dist_pd < 0,
                 "H4 RSI14 背離": lambda X: X.rsi_div > 5,
                 "H5 Hull MA 剛轉向": lambda X: X.hma_turn == 1,
                 "H6 量能高潮": lambda X: X.tv_e > 1.5}.items():
    X = nonoverlap(T3[np.asarray(fn(T3))])
    P(rline(name, X.pnl_R, X.entry_time))

# ---- 第五部分：M1 型態 + 背離 / 速度 / 多時框（rules5.RULES，門檻 = 前半分位數）----
from m1_momentum import candidate_table as m1_table
T5 = m1_table(BARS)
DEV = T5[T5.entry_time < CUT]
q = DEV.quantile(numeric_only=True, q=[0.2, 0.5, 0.8])
RULES5 = {
    "全部 M1 型態": lambda X: X.index == X.index,
    "V1 RSI 背離": lambda X: X.rsi_div > 0,
    "V2 MACD 柱背離": lambda X: X.macd_div > 0,
    "V3 速度背離": lambda X: X.vel_div > 0,
    "V1+V2": lambda X: (X.rsi_div > 0) & (X.macd_div > 0),
    "V1+V2+V3 三種背離": lambda X: (X.rsi_div > 0) & (X.macd_div > 0) & (X.vel_div > 0),
    "A2 背離 + 已減速": lambda X: (X.rsi_div > 0) & (X.macd_div > 0) & (X.acc3_e < 0),
    "V1+V2 + 順 M15": lambda X: (X.rsi_div > 0) & (X.macd_div > 0) & (X.align_15min == 1),
}
P("\n[第五部分] M1 型態 + 背離（2R / 60 分鐘、可重疊；黃金驗證期全部 −0.02 ~ −0.04R）")
for name, fn in RULES5.items():
    X = T5[np.asarray(fn(T5))]
    P(rline(name, X.pnl_R, X.entry_time))

# =====================================================================================================
P("\n" + "=" * 130 + "\n三、z-score / 速度")
P("\n[第五部分] M1 速度 / 加速度（vel = n 根變化 /(n × ATR60)、acc = vel 的變化；門檻 = 前半中位數）")
for name, fn in {"S3 最後 5 根速度不快（< 中位）": lambda X: X.vel5_e < q.loc[0.5, "vel5_e"],
                 "最後 5 根速度快（> 80%）": lambda X: X.vel5_e > q.loc[0.8, "vel5_e"],
                 "A1 已減速（acc3 < 0）": lambda X: X.acc3_e < 0,
                 "仍在加速（acc3 > 0）": lambda X: X.acc3_e > 0,
                 "S1+S2+S3（RSI 不極端、ER20 低、速度不快）": lambda X: (X.rsi_e > q.loc[0.5, "rsi_e"]) & (X.er_m1_20 < q.loc[0.5, "er_m1_20"]) & (X.vel5_e < q.loc[0.5, "vel5_e"])}.items():
    X = T5[np.asarray(fn(T5))]
    P(rline(name, X.pnl_R, X.entry_time))

# ---- 第四部分：型態 + z-score / 布林 / ER 比值（mr_features，可重疊，2R）----
from mr_features import candidate_table as mr_table
P("\n[第四部分] 型態 + z-score / 布林帶 / ER 比值（2R、可重疊；黃金 M15：外軌外收盤 +0.18R、三選二 +0.15R；z-score 不穩定）")
for tf, bars in (("M15", M15), ("M5", M5)):
    T4 = mr_table(bars)
    Bc, Cc, Dc = T4.er5_20 >= 2, T4.bbw_pct <= .5, T4.close_outside == 1
    for name, m in {"全部型態": T4.index == T4.index, "B ER5/ER20 ≥ 2": Bc, "C 帶寬 ≤ 50%": Cc, "D 外軌外收盤": Dc,
                    "三選二": (Bc.astype(int) + Cc.astype(int) + Dc.astype(int)) >= 2}.items():
        X = T4[np.asarray(m)]
        P(rline(f"{tf} {name}", X.pnl_R, X.entry_time))
    for z in ("z20", "z50", "z100"):
        e = np.nanquantile(T4[T4.entry_time < CUT][z], [0, .2, .4, .6, .8, 1]); e[0], e[-1] = -np.inf, np.inf
        g = pd.cut(T4[z], e)
        cells = [f"Q{i + 1} {T4[(g == c) & (T4.entry_time < CUT)].pnl_R.mean():+.2f}|{T4[(g == c) & (T4.entry_time >= CUT)].pnl_R.mean():+.2f}"
                 for i, c in enumerate(g.cat.categories)]
        P(f"  {tf} {z} 五分位（前半|後半，Q5 = 離均線最遠）  " + "  ".join(cells))

# ---- 第六部分：均線 / sigma 密集後突破（M1 中心參數）----
from squeeze_breakout import SqCfg, run as sq_run
P("\n[第六部分] 密集後突破（M1，EMA 5/10/20、下 20%、箱體 20、發散確認、箱體停損、3R；黃金 +0.04R，單純突破 −0.13/−0.02R）")
for comp, lab in (("ma", "均線密集"), ("sigma", "sigma（std20/ATR）密集"), ("none", "單純突破（對照）")):
    tr = sq_run(BARS, SqCfg(ma_type="ema", ma_lens=(5, 10, 20), comp=comp, pct=0.2, box=20, fan=True, stop="box", exit="3R", hold=240))
    P(rline(f"M1 {lab}", tr.pnl_R, tr.entry_time))

# =====================================================================================================
P("\n" + "=" * 130 + "\n四、其他（第三、八、九部分）")
import lonbrk
from h1_strategy import run_h1
from orb_fade import fade
m15s, m1s = M15.copy(), BARS.copy()
m15s["sp"] = m15s.spread * 0.01; m1s["sp"] = m1s.spread * 0.01
L = lonbrk.trades(m15s[["open", "high", "low", "close", "sp"]])
P(rline("LONBRK 倫敦開盤突破亞洲區間（M15）", L.pnl_R, L.entry_time) + "   黃金 2026 +0.20R；2023~25 −0.06R")
L1 = lonbrk.trades(m1s[["open", "high", "low", "close", "sp"]])
P(rline("LONBRK（M1 逐根）", L1.pnl_R, L1.entry_time))
H1 = run_h1(M15)
P(rline("H1 衝刺竭盡反轉（M15）", H1.pnl_R, H1.entry_time) + "   黃金 2026 +0.24R；2023~25 −0.03R")
for tg in (True, False):
    F = fade(m15s, 15, (15, 30), 15, mode="touch", target=tg)
    P(rline(f"紐約 OR15 假突破反向（{'目標 = 另一側' if tg else '持有到 21:00'}）", F.pnl_R, F.day) + "   黃金 2026 +0.12R；2023~25 −0.10R")
F = fade(m15s, 15, (16, 30), 15, mode="touch", target=True)
P(rline("（附）美股 16:30 OR15 假突破反向", F.pnl_R, F.day))
out.close()
