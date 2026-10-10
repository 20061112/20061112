"""第五十二部分 C：Range Bar 未來函數檢查 + 交易怎麼跑 + 逐筆交易統計。

1. 截斷測試：隨機抽事件時間 τ，把 M1 資料砍到 τ 那根為止，重新建磚塊、重算全部特徵，和用完整資料算的值比對。
   若有任何特徵用到 τ 之後的資料，兩者會不同。
2. 延後進場：訊號在第 i 根 M1 收盤出現，改成第 i+1 根「開盤」進場、出場也延後到下一個事件那根的下一根開盤。
   若結果靠的是同一根 K 棒內的資訊，延後後會明顯變差。
3. 交易規則（第五十二部分 A 的事件交易）逐筆統計，單位：美元/盎司（= 0.01 手的美元），每筆扣進場 K 棒點差。
"""
import numpy as np
import pandas as pd
from tpo import load_m1, build_days
from entropy_vol import build_bricks, brick_features, time_features
from part52_rangebar import build_events, fit_predict, MODELS, CUT, N_WIN

out = open("results_part52.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
RNG = np.random.default_rng(52)
FEATS = ["H_dir", "H_trans", "MI", "p_cont", "drift", "V_seas", "V_roll", "J60", "ER60", "PE"]


def feats_at(d, D, pos):
    """只用 d 的前 pos+1 根重算，回傳第 pos 根收盤時的特徵。"""
    dt = d.iloc[:pos + 1]
    B = build_bricks(dt, D.atr10.to_dict())
    F = brick_features(B, N_WIN).iloc[-1]
    TF = time_features(dt).iloc[-1]
    return {**{c: F[c] for c in ["H_dir", "H_trans", "MI", "p_cont", "drift"]},
            **{c: TF[c] for c in ["V_seas", "V_roll", "J60", "ER60", "PE"]}}, int(B.i.iloc[-1])


def trade_table(E, side, d, delay=False):
    """每個事件一筆：side（+1 做延續 / −1 做反轉）× 磚塊方向；持有到下一個事件。"""
    O = d.open.values; C = d.close.values; S = d.spread.values * 0.01
    i_in = E.i.values; i_out = np.r_[E.i.values[1:], -1]
    ok = i_out >= 0
    if delay:
        ok &= (i_out + 1 < len(O))
        px_in = O[np.minimum(i_in + 1, len(O) - 1)]; px_out = O[np.minimum(i_out + 1, len(O) - 1)]
    else:
        px_in = C[i_in]; px_out = C[np.maximum(i_out, 0)]
    direction = side * E.s.values
    T = pd.DataFrame(dict(t_in=d.index[i_in], t_out=d.index[np.maximum(i_out, 0)], dir=direction, brick=E["size"].values,
                          px_in=px_in, px_out=px_out, spread=S[i_in]))
    T["gross"] = direction * (px_out - px_in); T["pnl"] = T.gross - T.spread
    T["hold_min"] = (T.t_out - T.t_in).dt.total_seconds() / 60
    return T[ok].reset_index(drop=True)


def stats(T, lab):
    p = T.pnl; w, l = p[p > 0], p[p <= 0]
    eq = p.cumsum(); dd = (eq - eq.cummax()).min()
    days = T.t_in.dt.normalize().nunique()
    lose = (p <= 0).astype(int); streak = lose.groupby((lose != lose.shift()).cumsum()).sum().max()
    P(f"  {lab:34s} {len(T):5d}筆 {len(T) / days:5.1f}筆/天 勝率 {len(w) / len(p):5.1%} 均賺 {w.mean():+5.2f} 均賠 {l.mean():+5.2f} "
      f"每筆 毛 {T.gross.mean():+.3f} 淨 {p.mean():+.3f} PF {w.sum() / -l.sum():.2f} 總 {p.sum():+7.0f} 最大回撤 {dd:+6.0f} 最長連虧 {streak} "
      f"持有中位 {T.hold_min.median():.0f} 分 點差均 {T.spread.mean():.3f}")


if __name__ == "__main__":
    d = load_m1(); D, _ = build_days(d)
    B, E, *_ = build_events(d, D)
    Bf = brick_features(B, N_WIN); TF = time_features(d)

    P("\n" + "=" * 140 + "\n第五十二部分 C：Range Bar 未來函數檢查與交易明細")
    # ---- 1. 截斷測試
    pick = np.sort(RNG.choice(len(E), 40, replace=False))
    bad, maxdiff = 0, {c: 0.0 for c in FEATS}
    for k in pick:
        pos = int(E.i.iloc[k])
        f_trunc, last_i = feats_at(d, D, pos)
        j = np.searchsorted(Bf.i.values, pos, side="right") - 1
        full = {c: Bf[c].iloc[j] for c in ["H_dir", "H_trans", "MI", "p_cont", "drift"]}
        full.update({c: TF[c].iloc[pos] for c in ["V_seas", "V_roll", "J60", "ER60", "PE"]})
        diffs = {c: abs(f_trunc[c] - full[c]) for c in FEATS}
        for c in FEATS: maxdiff[c] = max(maxdiff[c], diffs[c] if np.isfinite(diffs[c]) else 0)
        bad += any(v > 1e-9 for v in diffs.values() if np.isfinite(v)) or last_i != pos
    P(f"  1. 截斷測試（{len(pick)} 個隨機事件，資料砍到訊號那根為止重算）：不一致 {bad} 筆；各特徵最大差異 "
      + "  ".join(f"{c} {v:.1e}" for c, v in maxdiff.items()))

    # ---- 2 / 3. 交易
    IS, OS = E[E.t < CUT], E[E.t >= CUT]
    po, _ = fit_predict(IS, OS, MODELS["M6 M4 + ex_cont + 路徑 ER60/sER60/sdrift"])
    side6 = np.where(po > 0, 1, -1)
    P("\n  2~3. 事件交易（6/1~10/8 樣本外；每個事件進場，持有到下一個事件；同時只有一筆；美元/盎司）")
    for delay in (False, True):
        tag = "（延後 1 根開盤進出）" if delay else "（訊號棒收盤進出）"
        stats(trade_table(OS, np.ones(len(OS)), d, delay), "一律做延續" + tag)
        stats(trade_table(OS, -np.ones(len(OS)), d, delay), "一律做反轉" + tag)
        stats(trade_table(OS, side6, d, delay), "M6 模型方向" + tag)
    T = trade_table(OS, side6, d)
    T["月"] = T.t_in.dt.to_period("M")
    P("  M6 逐月：" + "  ".join(f"{m}: {g.pnl.sum():+.0f}（{len(g)} 筆，PF {g.pnl[g.pnl > 0].sum() / -g.pnl[g.pnl <= 0].sum():.2f}）"
                              for m, g in T.groupby("月")))
    P(f"  M6 毛利合計 {T.gross.sum():+.0f}、點差合計 {-T.spread.sum():+.0f} → 淨 {T.pnl.sum():+.0f}")
    T.drop(columns="月").to_csv("part52_trades_M6.csv", index=False, float_format="%.3f")
    out.close()
