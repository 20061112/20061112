"""第三十五部分：三個濾網（反向尾巴、120 天方向開關、同方向最多 3 筆）的穩健性測試。
系統：L1 核心 → 尾巴（90% 分位，每月前推） → 120 天開關（用尾巴後的單） → 同方向同時最多 3 筆。
測試：
  1. 隨機對照（安慰劑）：每個濾網換成「刪掉一樣多、但隨機挑」的版本 300 次，看真的濾網排在第幾百分位
  2. 參數鄰近：尾巴分位、開關天數 × 門檻、疊單上限，各自掃一圈
  3. 逐年 / 逐半年：每個濾網的邊際貢獻是不是每段都正
  4. 日區塊 bootstrap：重抽交易日 1000 次，看「加濾網 − 不加」的 PF 差 > 0 的機率
  5. 成本壓力：每筆再多扣 0.02 / 0.05 / 0.1R
  6. 拿掉最好的 N 天
PF 以 R 計；ret15 = 調到平倉回撤 15% 時的總報酬。
"""
import numpy as np
import pandas as pd
from part29_dir_filter import L1, walk, dedup, OOS_END, pf
from part32_side_gate import DAYS
from part34_combo_stack import gate, cap

rng = np.random.default_rng(7)
out = open("results_part35.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")
    out.flush()


def dd(T):
    eq = T.sort_values("exit_time").R.cumsum()
    return (eq - eq.cummax()).min()


def m(T):
    d = dd(T)
    return dict(n=len(T), pf=pf(T.R), tot=T.R.sum(), dd=d, ret15=T.R.sum() * 15 / -d if d < 0 else np.nan)


def fmt(x):
    return f"{x['n']:5d} 筆 PF {x['pf']:.3f} 總 {x['tot']:+5.0f}R 回撤 {x['dd']:5.0f}R ret15 {x['ret15']:+4.0f}%"


C = dedup(L1)
S1 = walk(L1, 0.9, 1.0, 0.0)                    # + 尾巴
S2 = gate(S1, S1, 120)                           # + 開關
S3 = cap(S2, K=3)                                # + 疊單上限
STEPS = [("核心", C), ("+尾巴", S1), ("+開關", S2), ("+疊單≤3", S3)]
P("第三十五部分：三個濾網的穩健性測試（L1 核心 → 尾巴 → 120 天開關 → 同方向最多 3 筆）")
for lab, T in STEPS:
    P(f"  {lab:8s} {fmt(m(T))}")

# ---------- 1. 隨機對照 ----------
P("\n# 1. 隨機對照（300 次）：同樣刪掉那麼多，但隨機挑")
NR = 300
# 尾巴：每個月從核心隨機刪掉「和尾巴一樣多」的單
C2 = C.copy()
C2["ym"] = C2.day.dt.to_period("M")
kept = set(zip(S1.t_in, S1.side))
C2["k"] = [(t, s) in kept for t, s in zip(C2.t_in, C2.side)]
C2 = C2[C2.ym >= S1.day.dt.to_period("M").min()]
need = C2.groupby("ym").k.sum()
res = []
for _ in range(NR):
    parts = [g.sample(int(need[ym]), random_state=int(rng.integers(1e9))) for ym, g in C2.groupby("ym")]
    res.append(m(pd.concat(parts)))
R = pd.DataFrame(res)
a = m(S1)
P(f"  尾巴     真的 PF {a['pf']:.3f} ret15 {a['ret15']:.0f}% | 隨機 PF 中位 {R.pf.median():.3f}（5~95%：{R.pf.quantile(.05):.3f}~{R.pf.quantile(.95):.3f}），"
  f"真的勝過 {100 * (R.pf < a['pf']).mean():.0f}% | ret15 勝過 {100 * (R.ret15 < a['ret15']).mean():.0f}%")


# 開關：把每一邊的「開 / 關」序列整段平移隨機天數（保留關閉的天數和連續性，只是放在錯的時間）
def gate_mask(T, N=120):
    st = {}
    for s in (1, -1):
        daily = T[T.side == s].groupby("day").R.sum().reindex(DAYS, fill_value=0)
        st[s] = (daily.rolling(N, min_periods=N).sum().shift(1) < 0).to_numpy()
    return st


off = gate_mask(S1)
valid = ~np.isnan(S1.groupby("day").R.sum().reindex(DAYS).to_numpy()) | True
day_idx = pd.Series(np.arange(len(DAYS)), index=DAYS)
di = day_idx.reindex(S1.day).to_numpy()
res = []
for _ in range(NR):
    keep = np.ones(len(S1), bool)
    for s in (1, -1):
        sh = np.roll(off[s], int(rng.integers(60, len(DAYS) - 60)))
        keep &= ~((S1.side.to_numpy() == s) & sh[di])
    res.append(m(S1[keep]))
R = pd.DataFrame(res)
a = m(S2)
P(f"  開關     真的 PF {a['pf']:.3f} ret15 {a['ret15']:.0f}% | 隨機平移 PF 中位 {R.pf.median():.3f}（5~95%：{R.pf.quantile(.05):.3f}~{R.pf.quantile(.95):.3f}），"
  f"真的勝過 {100 * (R.pf < a['pf']).mean():.0f}% | ret15 勝過 {100 * (R.ret15 < a['ret15']).mean():.0f}%")

# 疊單：每天每方向保留「同樣筆數」但隨機挑（真的是保留最早的）
S2b = S2.copy()
cnt = S3.groupby(["day", "side"]).size()
res = []
for _ in range(NR):
    parts = []
    for (d_, s_), g in S2b.groupby(["day", "side"]):
        k = int(cnt.get((d_, s_), 0))
        if k:
            parts.append(g.sample(k, random_state=int(rng.integers(1e9))))
    res.append(m(pd.concat(parts)))
R = pd.DataFrame(res)
a = m(S3)
P(f"  疊單≤3   真的 PF {a['pf']:.3f} ret15 {a['ret15']:.0f}% | 同筆數隨機挑 PF 中位 {R.pf.median():.3f}（5~95%：{R.pf.quantile(.05):.3f}~{R.pf.quantile(.95):.3f}），"
  f"真的勝過 {100 * (R.pf < a['pf']).mean():.0f}% | ret15 勝過 {100 * (R.ret15 < a['ret15']).mean():.0f}%")
P("  （疊單上限的目的是降回撤、不是挑好單，所以 PF 和隨機差不多是預期中的；重點看 ret15）")

# ---------- 2. 參數鄰近 ----------
P("\n# 2. 參數鄰近")
P("  尾巴分位（後面接 120 天開關 + 最多 3 筆）")
for q in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 1.0):
    T1 = walk(L1, q, 1.0, 0.0) if q < 1 else C
    P(f"    q={q:.2f}: {fmt(m(cap(gate(T1, T1, 120), K=3)))}")
P("  開關天數 × 門檻（尾巴 0.9 + 最多 3 筆）；格子 = PF / ret15")
NS = (60, 80, 100, 120, 140, 160, 180, 200, 250, 300)
TH = (-10, -5, 0, 5)
P("    N    " + "  ".join(f"{t:>+12d}R" for t in TH))
grid = {}


def gate_thr(T, N, thr):
    keep = np.ones(len(T), bool)
    for s in (1, -1):
        daily = T[T.side == s].groupby("day").R.sum().reindex(DAYS, fill_value=0)
        v = daily.rolling(N, min_periods=N).sum().shift(1).reindex(T.day).to_numpy()
        keep &= ~((T.side.to_numpy() == s) & (v < thr))
    return T[keep]


for N in NS:
    cells = []
    for t in TH:
        x = m(cap(gate_thr(S1, N, t), K=3))
        grid[(N, t)] = x
        cells.append(f"{x['pf']:.2f}/{x['ret15']:+4.0f}%")
    P(f"    {N:3d}  " + "  ".join(f"{c:>13s}" for c in cells))
P(f"    不加開關：{fmt(m(cap(S1, K=3)))}")
P("  疊單上限 K（尾巴 0.9 + 120 天開關）")
for K in (1, 2, 3, 4, 5, 6, 8, None):
    P(f"    K={str(K) if K else '不限':>3s}: {fmt(m(cap(S2, K=K) if K else S2))}")

# ---------- 3. 逐年 / 逐半年邊際貢獻 ----------
P("\n# 3. 每一步的邊際貢獻（PF 差），逐年與逐半年")
h = lambda T: T.day.dt.year.astype(str) + np.where(T.day.dt.month <= 6, "H1", "H2")
for keyname, key in (("年", lambda T: T.day.dt.year.astype(str)), ("半年", h)):
    tb = pd.DataFrame({lab: T.groupby(key(T)).R.apply(pf) for lab, T in STEPS})
    P(f"\n  逐{keyname} PF")
    P(tb.round(2).to_string())
    diff = tb.diff(axis=1).iloc[:, 1:]
    P(f"  每一步 PF 變化 > 0 的期數：" + "  ".join(f"{c}: {(diff[c] > 0).sum()}/{diff[c].notna().sum()}" for c in diff.columns))
    rr = pd.DataFrame({lab: T.groupby(key(T)).R.sum() for lab, T in STEPS})
    P(f"  每期總 R：\n{rr.round(0).to_string()}")

# ---------- 4. 日區塊 bootstrap ----------
P("\n# 4. 日區塊 bootstrap（重抽交易日 1000 次）：每一步「加 − 不加」")
days = np.array(sorted(set(C.day[C.day >= S1.day.min()])))
agg = {lab: T.groupby("day").R.agg(win=lambda x: x[x > 0].sum(), loss=lambda x: -x[x <= 0].sum(), tot="sum").reindex(days, fill_value=0) for lab, T in STEPS}
B = 1000
idx = rng.integers(0, len(days), size=(B, len(days)))
pfs = {lab: a_.win.to_numpy()[idx].sum(1) / a_.loss.to_numpy()[idx].sum(1) for lab, a_ in agg.items()}
labs = [s[0] for s in STEPS]
for x, y in zip(labs[:-1], labs[1:]):
    d_ = pfs[y] - pfs[x]
    P(f"  {y:8s} vs {x:6s}: PF 差 中位 {np.median(d_):+.3f}（5~95%：{np.quantile(d_, .05):+.3f}~{np.quantile(d_, .95):+.3f}），> 0 的機率 {100 * (d_ > 0).mean():.0f}%")
d_ = pfs[labs[-1]] - pfs[labs[0]]
P(f"  全部三個 vs 核心: PF 差 中位 {np.median(d_):+.3f}（5~95%：{np.quantile(d_, .05):+.3f}~{np.quantile(d_, .95):+.3f}），> 0 的機率 {100 * (d_ > 0).mean():.0f}%")
for lab in labs:
    P(f"  {lab:8s} PF 5~95%：{np.quantile(pfs[lab], .05):.2f} ~ {np.quantile(pfs[lab], .95):.2f}，PF < 1 的機率 {100 * (pfs[lab] < 1).mean():.1f}%")

# ---------- 5. 成本壓力 ----------
P("\n# 5. 成本壓力：每筆再多扣 c R（滑價 / 手續費）")
for c in (0, 0.02, 0.05, 0.1, 0.15):
    P(f"  多扣 {c:.2f}R: " + " | ".join(f"{lab} PF {pf(T.R - c):.2f}" for lab, T in STEPS))
P(f"  （停損距離中位約 {S3.risk.median():.1f} 美元 → 0.05R ≈ {0.05 * S3.risk.median():.2f} 美元滑價）")

# ---------- 6. 拿掉最好的 N 天 ----------
P("\n# 6. 拿掉最好的 N 天（總 R）")
for n in (0, 5, 10, 20, 30):
    P(f"  拿掉 {n:2d} 天: " + " | ".join(f"{lab} {T.groupby('day').R.sum().sort_values(ascending=False).iloc[n:].sum():+5.0f}R" for lab, T in STEPS))
out.close()
