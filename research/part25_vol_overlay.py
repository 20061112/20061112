"""第二十五部分：把第二十二～二十四部分「能預測波動」的指標套到版本 A（不改進出場訊號，只改部位大小 / 停損寬度）。

版本 A 的交易：final_A.py 的 gen + walk_forward（滾動前推門檻，2026/3 ~ 10）。
波動預測（全部只用進場 K 棒收盤以前的 M1 資料）：
  vr   = 最近 5 根相對量 / 前 30 根相對量（量先於波動）
  dATR = ln ATR60 − ln ATR60(30 根前)（波動慣性）
  dn   = ln n(現價) − ln n(30 根前價位)，n = 過去 1 天剖面的停留密度（費馬介質；往稀薄區走 → 波動變大，所以取負號）
  綜合 C = 三者百分位的平均（百分位以 2025 全年 M1 為參考分布，交易期間 2026/3 以後，所以沒有前視）
所有變體事先列好、全部報告；評分以 R（固定風險比例）計，並分前後兩半（3~5 月 / 6~10 月）。
輸出：results_part25.txt
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from load import load
import physics_features as pf
from fermat_core import _fermat
from final_A import CFG, walk_forward
from part17_balance import gen
from part14_deep import ARR, DAYS

out = open("results_part25.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


# ------------------------------------------------------------------ 版本 A 交易
T, _ = walk_forward(gen(**CFG))
T = T.reset_index(drop=True)
P(f"版本 A：{len(T)} 筆，{T.day.min().date()} ~ {T.day.max().date()}，總 R {T.R.sum():+.1f}，PF {T.pnl[T.pnl > 0].sum() / -T.pnl[T.pnl <= 0].sum():.2f}")

# ------------------------------------------------------------------ 波動預測
d = pf.base(load("data/XAUUSD_M1_2023_2026.csv"))
vr = d.rvol.rolling(5).mean() / d.rvol.rolling(30).mean().shift(5)
datr = np.log(d.atr) - np.log(d.atr.shift(30))
pos = d.index.get_indexer(pd.DatetimeIndex(T.t_in))
assert (pos >= 0).all()
ref = np.where((d.index >= "2025-01-01") & (d.index < "2026-01-01"))[0][::5]
allpos = np.r_[ref, pos]
_, _, n_now, n_start = _fermat(d.high.values, d.low.values, d.close.values, d.atr.values, allpos.astype(np.int64), 30, 1440, 0.1, 1.0)
dn_all = np.log(n_now) - np.log(n_start)
dn_ref, dn_tr = dn_all[:len(ref)], dn_all[len(ref):]


def pct(ref_vals, x):
    r = np.sort(ref_vals[np.isfinite(ref_vals)])
    return np.searchsorted(r, x) / len(r)


T["vr"], T["datr"], T["dn"] = vr.values[pos], datr.values[pos], dn_tr
T["p_vr"] = pct(vr.values[ref], T.vr.values)
T["p_datr"] = pct(datr.values[ref], T.datr.values)
T["p_dn"] = 1 - pct(dn_ref, T.dn.values)
T["C"] = T[["p_vr", "p_datr", "p_dn"]].mean(1)
C_ref = (pct(vr.values[ref], vr.values[ref]) + pct(datr.values[ref], datr.values[ref]) + 1 - pct(dn_ref, dn_ref)) / 3
hiC = np.nanquantile(C_ref, 2 / 3)
loC = np.nanquantile(C_ref, 1 / 3)
T["Cg"] = np.where(T.C >= hiC, "高", np.where(T.C < loC, "低", "中"))
# 進場後實際波動（驗證預測有沒有對到）：進場後 60 根路徑長 / ATR60
path60 = d.ret.abs().rolling(60).sum().shift(-60) / d.atr
T["realvol60"] = path60.values[pos]
T["half"] = np.where(T.day < pd.Timestamp("2026-06-01"), "前半 3~5月", "後半 6~10月")

P(f"\n綜合 C 分組門檻（2025 參考分布）：低 < {loC:.3f} ≤ 中 < {hiC:.3f} ≤ 高")
P("\n1. 預測有沒有對到：依 C 分組，進場後 60 根實際波動、停損率、R")
g = T.groupby("Cg")
tab = pd.DataFrame({"筆數": g.size(), "實際波動60(ATR)": g.realvol60.mean(), "停損率": g.exit_reason.apply(lambda x: (x == "停損").mean()),
                    "平均R": g.R.mean(), "中位R": g.R.median(), "勝率": g.R.apply(lambda x: (x > 0).mean()),
                    "最大浮虧R中位": g.apply(lambda x: (x.mae / x.risk).median())}).loc[["低", "中", "高"]]
P(tab.round(3).to_string())
P("  各分量單獨（Spearman，與該筆 R / 實際波動）：")
for k in ("vr", "datr", "dn", "C"):
    P(f"    {k:5s}  vs R {T[k].corr(T.R, method='spearman'):+.3f}   vs 實際波動 {T[k].corr(T.realvol60, method='spearman'):+.3f}")
P("  前後兩半的平均 R（C 低 / 中 / 高）：")
P(T.pivot_table(index="half", columns="Cg", values="R", aggfunc="mean")[["低", "中", "高"]].round(3).to_string())


# ------------------------------------------------------------------ 重新模擬停損（與 final_system.simulate 相同邏輯，停損可改）
def resim(T, stop_px):
    R = np.empty(len(T)); why = []
    for n, (r, st) in enumerate(zip(T.itertuples(), stop_px)):
        t, O, H, L, C, S = ARR[r.day]
        side, lvl = r.side, r.lvl
        risk = side * (lvl - st)
        px, w = C[-1], "收盤"
        for k in range(r.i, len(C)):
            if (L[k] <= st) if side == 1 else (H[k] >= st):
                px = min(O[k], st) if side == 1 else max(O[k], st); w = "停損"; break
        R[n] = (side * (px - lvl) - r.spread) / risk
        why.append(w)
    return R, why


base_stop = np.where(T.side == 1, T.ibl, T.ibh).astype(float)
R0, _ = resim(T, base_stop)
assert np.allclose(R0, T.R.values, atol=1e-9), "重新模擬與原版不一致"
wide = base_stop - T.side.values * 0.5 * T.rng.values
mid = ((T.ibh + T.ibl) / 2).values
R_wide, _ = resim(T, wide)
R_mid, _ = resim(T, mid)
high = (T.Cg == "高").values
low = (T.Cg == "低").values

V = {
    "S0 原版 A": (T.R.values, np.ones(len(T))),
    "S1 C 高 → 半倉": (T.R.values, np.where(high, 0.5, 1.0)),
    "S2 C 高 → 不做": (T.R.values, np.where(high, 0.0, 1.0)),
    "S3 連續：倉位 = 1.5 − C": (T.R.values, 1.5 - T.C.values),
    "S4 反向：C 高 → 1.5 倍倉": (T.R.values, np.where(high, 1.5, 1.0)),
    "S5 C 低 → 半倉": (T.R.values, np.where(low, 0.5, 1.0)),
    "T1 C 高 → 停損放寬 0.5×IB": (np.where(high, R_wide, T.R.values), np.ones(len(T))),
    "T2 全部停損放寬 0.5×IB（對照）": (R_wide, np.ones(len(T))),
    "T3 C 低 → 停損收到 IB 中點": (np.where(low, R_mid, T.R.values), np.ones(len(T))),
    "T4 全部停損收到 IB 中點（對照）": (R_mid, np.ones(len(T))),
}


def evaluate(R, w, idx=None):
    X = pd.DataFrame({"r": R * w, "w": w, "day": T.day, "exit": T.exit_time})
    if idx is not None:
        X = X[idx]
    X = X[X.w > 0].sort_values("exit")
    eq = X.r.cumsum()
    dd = (eq - eq.cummax()).min()
    days = [dd_ for dd_ in DAYS if X.day.min() <= dd_ <= X.day.max() and dd_.dayofweek != 0]
    daily = pd.Series(0.0, index=days).add(X.groupby("day").r.sum(), fill_value=0)
    gp, gl = X.r[X.r > 0].sum(), -X.r[X.r < 0].sum()
    return {"筆數": len(X), "總R": X.r.sum(), "PF": gp / gl, "最大回撤R": dd, "總R/回撤": X.r.sum() / -dd,
            "日Sharpe": daily.mean() / daily.std() * np.sqrt(250), "平均倉位": X.w.mean()}


P("\n2. 部位 / 停損變體（R 單位 = 每筆固定風險；倉位 0.5 = 該筆只冒一半風險）")
rows = {k: evaluate(*v) for k, v in V.items()}
P(pd.DataFrame(rows).T.round(2).to_string())
P("\n3. 前後兩半（總R / 回撤 / Sharpe）")
for h in ("前半 3~5月", "後半 6~10月"):
    m = (T.half == h).values
    rr = pd.DataFrame({k: evaluate(*v, idx=m) for k, v in V.items()}).T[["總R", "最大回撤R", "總R/回撤", "日Sharpe"]]
    P(f"  {h}")
    P(rr.round(2).to_string())
P("\n4. 分量單獨做 S1（C 換成單一指標的前 1/3 → 半倉）")
for k, lab in (("p_vr", "量比"), ("p_datr", "ATR 變化"), ("p_dn", "費馬 dn（往稀薄區）")):
    h1 = T[k].values >= 2 / 3
    e = evaluate(T.R.values, np.where(h1, 0.5, 1.0))
    P(f"  {lab:14s} 總R {e['總R']:+.1f}  回撤 {e['最大回撤R']:.1f}  總R/回撤 {e['總R/回撤']:.2f}  Sharpe {e['日Sharpe']:.2f}")
out.close()
