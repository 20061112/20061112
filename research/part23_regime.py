"""第二十三部分 第 4 步：先判斷平衡 / 失衡，再看參考價與 OR 突破的反應（條件式規則）。

情境（全部在當天開盤前就知道）：
  va_rel  前一天價值區 vs 前兩天價值區：重疊 ≥ 50%（以較窄者計）→ balance；否則依中點高低 → up / down（價值移動）
  bracket 前 3 天合併區間 / 前 3 天平均日區間 ≤ 1.6 → 多日平衡區間（bracket），否則 → 非 bracket
  pday    前一天的日型：|收 − 開| / 區間 ≥ 0.6 → 趨勢日、≤ 0.3 → 平衡日
事前登記的預測（AMT）：
  R1  平衡情境（balance / bracket）：參考價（前日高低、前一 session 高低、VA 邊緣）反彈率 > 失衡情境（回應型）
  R2  價值移動（up / down）：順移動方向去碰的參考價（往上移時碰上方的高點、VAH）被突破較多；逆方向的較會撐住
  R3  OR 突破：平衡情境下延續較差、失衡且順價值移動方向的突破延續較好
資料：part23_events.csv（第 1 步的觸碰事件）、part22_breakout.csv（OR 突破事件）。
"""
import pickle
import numpy as np
import pandas as pd
from part22_core import tstat, SESS_ORDER

YRS = (2023, 2024, 2025, 2026)
D, S = pickle.load(open("part22_sessions.pkl", "rb"))

# ---- 情境
D = D.copy()
lo_ov = np.maximum(D.val, D.val.shift(1))
hi_ov = np.minimum(D.vah, D.vah.shift(1))
narrow = np.minimum(D.vah - D.val, (D.vah - D.val).shift(1))
ov = ((hi_ov - lo_ov).clip(lower=0) / narrow)
mid, mid1 = (D.vah + D.val) / 2, ((D.vah + D.val) / 2).shift(1)
rel = np.where(ov >= 0.5, "balance", np.where(mid > mid1, "up", "down"))
rng3 = D.h.rolling(3).max() - D.l.rolling(3).min()
avg3 = (D.h - D.l).rolling(3).mean()
eff = (D.c - D.o).abs() / (D.h - D.l)
Rg = pd.DataFrame({"va_rel": pd.Series(rel, index=D.index).shift(1),           # 用前一天與前兩天 → shift 1
                   "bracket": (rng3 / avg3 <= 1.6).shift(1),
                   "pday": pd.Series(np.where(eff >= 0.6, "trend", np.where(eff <= 0.3, "balance", "normal")), index=D.index).shift(1)})
Rg.index = Rg.index.astype(str)
Rg = Rg.iloc[3:]

E = pd.read_csv("part23_events.csv")
E = E.merge(Rg, left_on="day", right_index=True)
B = pd.read_csv("part22_breakout.csv")
B = B[B.kind == "real"].merge(Rg, left_on="day", right_index=True)


def br(g):
    x = g.b10[g.b10 != 0]
    return (x > 0).mean() * 100 if len(x) else np.nan


def dline(g, label):
    re, pl = g[g.kind == "real"], g[g.kind == "placebo"]
    yr = " ".join(f"{br(re[re.year == y]) - br(pl[pl.year == y]):+5.1f}" for y in YRS)
    a = br(re[re.year <= 2024]) - br(pl[pl.year <= 2024])
    b = br(re[re.year >= 2025]) - br(pl[pl.year >= 2025])
    return (f"  {label:<40} n={len(re):5d}  真 {br(re):5.1f}%  假 {br(pl):5.1f}%  差 {br(re) - br(pl):+5.1f}  "
            f"發現 {a:+5.1f} / 驗證 {b:+5.1f}  逐年 {yr}")


out = ["=" * 120, "第 4 步：平衡 / 失衡情境下的參考價反應（反彈率 真 − 假；正 = 撐住、負 = 被突破）", "=" * 120]
out.append("  情境分佈（天）: " + ", ".join(f"{k}={v}" for k, v in Rg.va_rel.value_counts().items()) +
           f"；bracket {Rg.bracket.mean()*100:.0f}%；前日趨勢日 {(Rg.pday == 'trend').mean()*100:.0f}% / 平衡日 {(Rg.pday == 'balance').mean()*100:.0f}%")
X = E[E.level.isin(["H", "L", "VAH", "VAL"])]
out.append("\n  [R1] 平衡 vs 失衡（所有高低點 + VA 邊緣）")
for col, vals in (("va_rel", ["balance", "up", "down"]), ("bracket", [True, False]), ("pday", ["balance", "normal", "trend"])):
    for v in vals:
        out.append(dline(X[X[col] == v], f"{col} = {v}"))
out.append("\n  [R1] 依 session")
for s in SESS_ORDER:
    for col, v in (("va_rel", "balance"), ("va_rel", "up_down"), ("bracket", True), ("bracket", False)):
        g = X[(X.sess == s) & ((X[col] == v) if v != "up_down" else X[col].isin(["up", "down"]))]
        out.append(dline(g, f"{s} {col} = {v}"))

# R2：順 / 逆價值移動方向。side = −1 表示從下方往上碰（往上越過）
X2 = X[X.va_rel.isin(["up", "down"])].copy()
mig = np.where(X2.va_rel == "up", 1, -1)
X2["with"] = (-X2.side) == mig
out.append("\n  [R2] 價值移動中：順移動方向去碰（往上移時由下往上碰）vs 逆方向")
for w in (True, False):
    out.append(dline(X2[X2["with"] == w], f"{'順' if w else '逆'}價值移動方向"))
    for s in SESS_ORDER:
        out.append(dline(X2[(X2["with"] == w) & (X2.sess == s)], f"  {s} {'順' if w else '逆'}"))
    for lv, nm in ((["H", "L"], "高低點"), (["VAH", "VAL"], "VA 邊緣")):
        out.append(dline(X2[(X2["with"] == w) & X2.level.isin(lv)], f"  {nm} {'順' if w else '逆'}"))

# R3：OR 突破
out += ["", "=" * 120, "  [R3] OR 突破 × 情境（抱到 session 結束，ATR，扣點差）", "=" * 120]


def bline(g, label):
    yr = " ".join(f"{g[g.year == y].rS.mean():+.3f}" for y in YRS)
    return (f"  {label:<40} n={len(g):5d}  rS {g.rS.mean():+.3f} (t {tstat(g.rS):+4.1f})  "
            f"發現 {g[g.year <= 2024].rS.mean():+.3f} / 驗證 {g[g.year >= 2025].rS.mean():+.3f}  逐年 {yr}")


for col, vals in (("va_rel", ["balance", "up", "down"]), ("bracket", [True, False]), ("pday", ["balance", "normal", "trend"])):
    for v in vals:
        out.append(bline(B[B[col] == v], f"{col} = {v}"))
B2 = B[B.va_rel.isin(["up", "down"])]
w = B2.dir == np.where(B2.va_rel == "up", 1, -1)
out.append(bline(B2[w], "順價值移動方向突破"))
out.append(bline(B2[~w], "逆價值移動方向突破"))
for s in SESS_ORDER:
    out.append(bline(B2[w & (B2.sess == s)], f"  {s} 順價值移動"))
    out.append(bline(B2[~w & (B2.sess == s)], f"  {s} 逆價值移動"))
    out.append(bline(B[(B.va_rel == "balance") & (B.sess == s)], f"  {s} 平衡（VA 重疊）"))

txt = "\n".join(out)
print(txt)
open("results_part23_step4.txt", "w").write(txt + "\n")
