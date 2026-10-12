"""第五十二部分補充：σ̂ 停損為什麼讓 2023~2024 變差？（相關 0.88 為什麼還差這麼多）"""
import numpy as np
import pandas as pd
import part52_user_entropy as U

ev, sig_ev, ok = U.ev, U.sig_ev, U.ok_ev
k = U.k_sig
t = pd.DatetimeIndex([U.IDX[e[0]] for e in ev])
atr_stop = np.array([2 * e[3] for e in ev])
sig_stop = k * sig_ev
ratio = sig_stop / atr_stop
D = pd.DataFrame({"t": t, "ratio": ratio, "ok": ok, "year": t.year, "hour": t.hour})
D = D[D.ok]
lines = ["σ̂ 停損 / ATR 停損 的比值（> 1 = σ̂ 停損較寬）"]
lines.append("  逐年中位數：" + "  ".join(f"{y}: {g.ratio.median():.2f}（10%~90%：{g.ratio.quantile(.1):.2f}~{g.ratio.quantile(.9):.2f}）" for y, g in D.groupby("year")))
lines.append("  時段中位數：" + "  ".join(f"{a:02d}-{b:02d}點 {D[D.hour.between(a, b - 1)].ratio.median():.2f}" for a, b in [(1, 4), (4, 8), (8, 11), (11, 15), (15, 18), (18, 22)]))
lines.append(f"  log 相關 0.88，但單筆比值 10%~90% 分位 {D.ratio.quantile(.1):.2f}~{D.ratio.quantile(.9):.2f} → 個別單的停損仍可能差 30~40%")

# 逐年：兩個版本的每筆 R、停損率、筆數、與原版重疊
O, S = U.T, U.T_sig
lines.append("\n逐年比較（實際下單）")
lines.append(f"  {'年':4s} {'原版筆數':>7s} {'σ̂版筆數':>7s} {'重疊':>5s} | {'原版每筆R':>8s} {'σ̂版每筆R':>8s} | {'原版停損率':>8s} {'σ̂版停損率':>8s} | {'原版$':>7s} {'σ̂版$':>7s}")
ko = set(zip(O.進場時間, O.dir)); ks = set(zip(S.進場時間, S.dir))
for y in (2023, 2024, 2025, 2026):
    o, s = O[O.進場時間.dt.year == y], S[S.進場時間.dt.year == y]
    ov = len(set(zip(o.進場時間, o.dir)) & set(zip(s.進場時間, s.dir)))
    lines.append(f"  {y} {len(o):7d} {len(s):7d} {ov / max(len(o), 1):5.0%} | {o.R.mean():+8.3f} {s.R.mean():+8.3f} | "
                 f"{(o.R < -0.9).mean():8.0%} {(s.R < -0.9).mean():8.0%} | {o.net.sum() * 2:+7,.0f} {s.net.sum() * 2:+7,.0f}")

# 拆解：同一批單（只看兩版都有的單）在 2023~2024 的差異 → 停損本身的影響；其餘 → 影子濾網選單不同的影響
both = pd.merge(O, S, on=["進場時間", "dir"], suffixes=("_o", "_s"))
b24 = both[both.進場時間 < "2025-01-01"]
only_o = O[(O.進場時間 < "2025-01-01") & ~pd.Series(list(zip(O.進場時間, O.dir))).isin(ks).values]
only_s = S[(S.進場時間 < "2025-01-01") & ~pd.Series(list(zip(S.進場時間, S.dir))).isin(ko).values]
lines.append("\n2023/8~2024/12 拆解（0.02 口美元）")
lines.append(f"  兩版都有的單 {len(b24)} 筆：原版 {b24.net_o.sum() * 2:+,.0f}、σ̂版 {b24.net_s.sum() * 2:+,.0f} → 停損寬度本身的影響 {(b24.net_s.sum() - b24.net_o.sum()) * 2:+,.0f}")
lines.append(f"  只有原版有的單 {len(only_o)} 筆 {only_o.net.sum() * 2:+,.0f}；只有σ̂版有的單 {len(only_s)} 筆 {only_s.net.sum() * 2:+,.0f} → 影子濾網選到不同的單 {(only_s.net.sum() - only_o.net.sum()) * 2:+,.0f}")
r_tight = b24[b24.risk_s < 0.8 * b24.risk_o]; r_wide = b24[b24.risk_s > 1.25 * b24.risk_o]
lines.append(f"  兩版都有的單中，σ̂ 停損明顯較窄（< 0.8 倍）{len(r_tight)} 筆：原版 {r_tight.net_o.sum() * 2:+,.0f} → σ̂版 {r_tight.net_s.sum() * 2:+,.0f}；"
             f"明顯較寬（> 1.25 倍）{len(r_wide)} 筆：{r_wide.net_o.sum() * 2:+,.0f} → {r_wide.net_s.sum() * 2:+,.0f}")
print("\n".join(lines))
open("results_part52_why.txt", "w").write("\n".join(lines) + "\n")
