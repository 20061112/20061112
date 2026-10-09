"""第十九部分：版本 A 的報酬來源 —— beta（方向曝險）、風格因子（日內趨勢 / 波動）、還是 alpha？"""
import numpy as np, pandas as pd
from part14_deep import ARR, D, DAYS
from part17_balance import gen
from final_A import CFG, walk_forward

out = open("results_part19.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

T, _ = walk_forward(gen(**CFG))
days = [d for d in DAYS if T.day.min() <= d <= T.day.max() and d.dayofweek != 0]
pnl = pd.Series(0.0, index=days).add(T.groupby("day").pnl.sum(), fill_value=0)
G = pd.DataFrame({d: dict(ret=ARR[d][4][-1] - ARR[d][1][0], rng=ARR[d][2].max() - ARR[d][3].min(), atr=D.atr10[d]) for d in days}).T
G["pnl"] = pnl
# 簡單日內順勢基準：11:00 依「開盤以來方向」進場、抱到收盤
def tsmom(d, h=11):
    t, O, H, L, C, S = ARR[d]; k = np.searchsorted(t, d + pd.Timedelta(hours=h))
    return np.sign(C[k] - O[0]) * (C[-1] - C[k]) if k < len(C) else 0.0
G["tsmom"] = [tsmom(d) for d in days]
G["abs_ret"] = G.ret.abs()

def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; s2 = e @ e / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, b / se, 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))

y = G.pnl.values
P("第十九部分：報酬來源拆解（版本 A，滾動前推 3~10 月，日損益，每筆 1 盎司）")
P(f"  交易日 {len(G)}；黃金這段期間 開盤→收盤 日報酬平均 {G.ret.mean():+.2f} 美元，策略日均 {G.pnl.mean():+.2f}")
b, t, r2 = ols(y, [G.ret]); P(f"\n1) 對黃金日報酬（方向 beta）：beta {b[1]:+.3f}（t {t[1]:+.2f}），截距 {b[0]:+.1f}/天（t {t[0]:+.2f}），R² {r2:.3f}")
b, t, r2 = ols(y, [G.abs_ret]); P(f"2) 對 |日報酬|（波動 / 趨勢日曝險）：係數 {b[1]:+.3f}（t {t[1]:+.2f}），截距 {b[0]:+.1f}（t {t[0]:+.2f}），R² {r2:.3f}")
b, t, r2 = ols(y, [G.tsmom]); P(f"3) 對簡單日內順勢（11:00 順開盤方向抱到收盤）：係數 {b[1]:+.3f}（t {t[1]:+.2f}），截距 {b[0]:+.1f}（t {t[0]:+.2f}），R² {r2:.3f}；"
                                f"相關 {np.corrcoef(y, G.tsmom)[0, 1]:.2f}；基準本身日均 {G.tsmom.mean():+.1f}（t {G.tsmom.mean() / G.tsmom.std() * np.sqrt(len(G)):+.2f}）")
b, t, r2 = ols(y, [G.ret, G.abs_ret, G.tsmom])
P(f"4) 三個一起：beta {b[1]:+.3f}（t {t[1]:+.2f}）、|報酬| {b[2]:+.3f}（t {t[2]:+.2f}）、順勢 {b[3]:+.3f}（t {t[3]:+.2f}）；"
  f"**截距（扣掉以上曝險後剩下的）{b[0]:+.1f}/天（t {t[0]:+.2f}）**，R² {r2:.3f}")

P("\n5) 多空拆開：" + "  ".join(f"{lab} {len(x)}筆 每筆 {x.pnl.mean():+.2f} PF {x.pnl[x.pnl > 0].sum() / -x.pnl[x.pnl <= 0].sum():.2f}"
                                 for lab, x in [("多", T[T.side == 1]), ("空", T[T.side == -1])]))
up, dn = G[G.ret > 0], G[G.ret < 0]
P(f"   黃金上漲日 {len(up)} 天 策略日均 {up.pnl.mean():+.1f}；下跌日 {len(dn)} 天 策略日均 {dn.pnl.mean():+.1f}")

# 6) 隨機方向：同樣的進場時間、同樣的停損距離（對稱放）、收盤出場
rng = np.random.default_rng(0)
def sim_side(r, side):
    t, O, H, L, C, S = ARR[r.day]; e = r.lvl; risk = r.risk; st = e - side * risk
    seg_l, seg_h = L[r.i:], H[r.i:]
    hit = np.where((seg_l <= st) if side == 1 else (seg_h >= st))[0]
    px = st if len(hit) else C[-1]
    return side * (px - e) - r.spread
same = np.array([sim_side(r, r.side) for r in T.itertuples()])
opp = np.array([sim_side(r, -r.side) for r in T.itertuples()])
rand_means = [np.where(rng.random(len(T)) < .5, same, opp).mean() for _ in range(5000)]
P(f"\n6) 同樣進場時間、同樣停損距離：")
P(f"   照策略方向 每筆 {same.mean():+.2f}；反方向 每筆 {opp.mean():+.2f}；隨機方向 每筆 {np.mean(rand_means):+.2f}"
  f"（95% 區間 {np.percentile(rand_means, 2.5):+.2f} ~ {np.percentile(rand_means, 97.5):+.2f}）")
P(f"   → 策略方向勝過 {(np.array(rand_means) < same.mean()).mean() * 100:.1f}% 的隨機方向")

# 7) 依日波幅分組
G["rng_atr"] = G.rng / G.atr
q = pd.qcut(G.rng_atr, 4, labels=["最小 25%", "25~50%", "50~75%", "最大 25%"])
P("\n7) 依當天波幅（/ATR）分組的策略日損益：" + "  ".join(f"{k}: {v:+.1f}" for k, v in G.groupby(q, observed=True).pnl.mean().items()))
out.close()
