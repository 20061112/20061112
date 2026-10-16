"""照 optimistic-meitner 第五十九部分的做法：把 ATR 原地換成波動預測 σ̂，其他規則、門檻一律不動。

原策略的波動訊號 = z( ln ATR%20 )，252 日回看，門檻 0.5。
替換版 = z( ln σ̂_h )，σ̂_h = 預測「未來 h 天」的日波動（har_rel 的日線版），同樣 252 日回看、同樣門檻 0.5。
（part59 的換算常數 c = 平均 TR / 平均 σ̂ 在取 log 後的 z-score 裡會抵銷，不影響訊號。）

har_rel 日線版（對應 volfc.py / part55_vol_verify.py）：
  y = ln(未來 h 天平均 RV) − ln RV_5d
  X = [ln RV_1d − ln RV_5d, ln RV_22d − ln RV_5d, ln RV_66d − ln RV_5d, 跳躍占比 J20]
  逐月 walk-forward（訓練樣本的目標期間必須在測試月之前結束），Duan smearing。
RV 兩種量法：TR（(真實區間 / 前收)²，最接近 ATR）、PK（Parkinson + 隔夜跳空²）。
σ̂ 缺值（前 3 年暖機）時退回原本的 ATR z。
輸出 results_sigma.txt、results_sigma.csv。
"""
import itertools
import numpy as np
import pandas as pd
import indicators as ind
import engine as E
from run_tests import q, OK0, Z, ABOVE, w_rule, evaluate


def rv_measure(d, kind):
    pc = d.close.shift(1)
    if kind == "TR":
        tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
        rv = (tr / pc) ** 2
    else:
        rv = np.log(d.high / d.low) ** 2 / (4 * np.log(2)) + np.log(d.open / pc) ** 2
    return rv.clip(lower=1e-10)


def har_rel_daily(d, h=5, kind="TR", warm_days=750):
    rv = rv_measure(d, kind)
    r = np.log(d.close).diff()
    l1 = np.log(rv); l5 = np.log(rv.rolling(5).mean())
    l22 = np.log(rv.rolling(22).mean()); l66 = np.log(rv.rolling(66).mean())
    J = ind.jump_ratio(r, 20)
    X = np.column_stack([l1 - l5, l22 - l5, l66 - l5, J])
    fut = np.log(rv[::-1].rolling(h).mean()[::-1].shift(-1))          # t+1..t+h
    y = (fut - l5).values
    l5v = l5.values
    n = len(d); out = np.full(n, np.nan)
    mon = d.index.to_period("M")
    ok = np.isfinite(X).all(1)
    for m in mon.unique():
        te = np.flatnonzero(mon == m)
        if te[0] < warm_days:
            continue
        tr = np.zeros(n, bool); tr[:max(te[0] - h, 0)] = True       # 目標期間在測試月前結束
        tr &= ok & np.isfinite(y)
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        sm = np.mean(np.exp(y[tr] - A @ b))
        tt = te[ok[te]]
        out[tt] = np.exp(np.column_stack([np.ones(len(tt)), X[tt]]) @ b + l5v[tt]) * sm
    return pd.Series(np.sqrt(out), index=d.index)        # 日波動（報酬單位）


def sig_z(d, sig, look=252):
    return ind.rz(np.log(sig), look)


def forecast_quality(d, sig, h, kind):
    """樣本外預測力：ln σ̂² 對 ln 實現（未來 h 天平均 RV）的 R²，與「直接用過去 20 日 ATR%²」比較。"""
    rv = rv_measure(d, kind)
    fut = np.log(rv[::-1].rolling(h).mean()[::-1].shift(-1))
    pc = d.close.shift(1)
    tr = np.maximum(d.high - d.low, np.maximum((d.high - pc).abs(), (d.low - pc).abs()))
    atr2 = np.log((tr.rolling(20).mean() / d.close) ** 2)
    m = np.isfinite(fut) & np.isfinite(np.log(sig ** 2)) & np.isfinite(atr2)
    r2 = lambda x: np.corrcoef(x[m], fut[m])[0, 1] ** 2
    return r2(np.log(sig ** 2)), r2(atr2)


if __name__ == "__main__":
    out = []
    P = lambda s: (print(s, flush=True), out.append(s))
    w0 = w_rule(OK0)
    b, broll, bcal = evaluate(w0)
    b, broll, bcal = evaluate(w0, broll, bcal)
    rows = [dict(name="BASE ATR z", h=0, kind="", th=0.5) | b]
    P(f"BASE  irr99 {b['irr99']:.3f} irr10 {b['irr10']:.3f} worst {b['worst']:.2f} g10min {b['g10_min']:.2f} "
      f"worstyr {b['worst_year']:.2f} sh {b['sharpe']:.2f} sw {b['sw']:.1f}")
    P("\n預測力（樣本外 R²：ln σ̂² vs ln 未來 h 天 RV；括號 = 過去 20 日 ATR%²）")
    sigs = {}
    for kind, h in itertools.product(["TR", "PK"], [1, 5, 10, 20]):
        s = har_rel_daily(q, h, kind)
        sigs[(kind, h)] = s
        a, bb = forecast_quality(q, s, h, kind)
        P(f"  {kind} h={h:2d}: R² {a:.3f}（ATR {bb:.3f}）")
    P("\n替換 ATR z → σ̂ z（MA200、門檻、回看都不動；σ̂ 缺值退回 ATR z）")
    for (kind, h), s in sigs.items():
        zs = sig_z(q, s).fillna(Z)
        for th in [0.25, 0.5, 0.75]:
            w = w_rule(ABOVE & (zs < th))
            o, _, _ = evaluate(w, broll, bcal)
            rows.append(dict(name="SIG", h=h, kind=kind, th=th) | o)
            P(f"  SIG {kind} h={h:2d} z<{th:<4} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} lev {o['lev']:.2f} "
              f"win10 {o['win10']:2d}/17 dIRR10y {o['d_irr10y_med']:+.3f}(min {o['d_irr10y_min']:+.3f}) worst {o['worst']:.2f} "
              f"g10min {o['g10_min']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f} sw {o['sw']:.1f}")
        # ATR z 與 σ̂ z 平均
        zs2 = ((Z + sig_z(q, s)) / 2).fillna(Z)
        w = w_rule(ABOVE & (zs2 < 0.5))
        o, _, _ = evaluate(w, broll, bcal)
        rows.append(dict(name="MIX", h=h, kind=kind, th=0.5) | o)
        P(f"  MIX {kind} h={h:2d} z<0.5  irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']:2d}/17 "
          f"dIRR10y {o['d_irr10y_med']:+.3f} worst {o['worst']:.2f} worstyr {o['worst_year']:.2f} sh {o['sharpe']:.2f}")
    # 對照：ATR 門檻本身的鄰域（看 σ̂ 的差異是否超出「只改門檻」的範圍）
    for th in [0.25, 0.75]:
        o, _, _ = evaluate(w_rule(ABOVE & (Z < th)), broll, bcal)
        rows.append(dict(name="ATR", h=0, kind="", th=th) | o)
        P(f"  ATR z<{th}           irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']:2d}/17 dIRR10y {o['d_irr10y_med']:+.3f}")
    R = pd.DataFrame(rows)
    R.to_csv("results_sigma.csv", index=False, float_format="%.4f")
    S = R[R.name == "SIG"]
    P(f"\nσ̂ 替換整片網格（{len(S)} 組）：irr99 中位 {S.irr99.median():.3f}（原 {b['irr99']:.3f}）、"
      f"10 年 IRR 差中位 {S.d_irr10y_med.median():+.3f}、勝過原版 IRR 的比例 {(S.irr99 > b['irr99']).mean():.0%}")
    S5 = S[S.th == 0.5]
    P(f"只看門檻 0.5（8 組）：irr99 中位 {S5.irr99.median():.3f}、10 年 IRR 差中位 {S5.d_irr10y_med.median():+.3f}、"
      f"10 年勝出段數中位 {S5.win10.median():.0f}/17")
    open("results_sigma.txt", "w").write("\n".join(out) + "\n")
