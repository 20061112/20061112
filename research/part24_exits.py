"""第二十四部分：M1 急跌買的出場方式（停損 ATR × 停利 RR × 最長持有 + 保本 / 移動停損）。

進場與 final_m1dip.py 完全相同：z_v20(1380) 首次 ≤ −2.5 → 下一根開盤做多；同時只持有一筆。
出場（逐根 M1）：
  停損 = 進場價 − SL × ATR60；停利 = 進場價 + RR × 停損距離（RR=0 表示不設停利）；
  同一根同時碰到停損與停利 → 一律當作先停損（保守）；跳空穿過 → 以開盤價成交；
  最長持有 HOLD 根，到時收盤平倉。
  變體 BE：浮盈達 1R 後（以該根最高價判斷），從下一根起停損移到進場價。
  變體 TR：移動停損 = 前一根為止的最高價 − SL × ATR60（只上移）。
成本：進場點差。單位：bp（÷ 進場價）、R（÷ 停損距離）。樣本內 < 2025-01-01，樣本外 ≥ 2025-01-01。
輸出：results_part24.txt、part24_grid.csv、part24_equity.png
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from load import load

THR, ZW = 2.5, 1380
SLS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
RRS = (0, 0.5, 1.0, 1.5, 2.0, 3.0)
HOLDS = (24, 48, 120, 240)
MODES = ("固定", "保本1R", "移動停損")
SPLIT = pd.Timestamp("2025-01-01")
MAXH = max(HOLDS)
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=6)
v = c.diff().where(~gap).ewm(span=20, adjust=False).mean() / atr
z = (v - v.rolling(ZW).mean()) / v.rolling(ZW).std()
o, h, l, cl, A, Z = m.open.to_numpy(), m.high.to_numpy(), m.low.to_numpy(), c.to_numpy(), atr.to_numpy(), z.to_numpy()
spr = m.spread.to_numpy() * 0.01
ts = m.index.to_numpy()

trig = Z <= -THR
cand = np.flatnonzero(trig & ~np.r_[False, trig[:-1]])
cand = cand[(cand + MAXH + 1 < len(m)) & np.isfinite(A[cand])]
# 每個候選的路徑（進場那根起 MAXH 根），以及到第幾根前不跨休市
E = cand + 1
idx = E[:, None] + np.arange(MAXH)[None, :]
PO, PH, PL, PC = o[idx], h[idx], l[idx], cl[idx]
span_ok = (ts[idx] - ts[cand][:, None]) <= (np.arange(MAXH)[None, :] + 1) * np.timedelta64(1, "m") * 1.6 + np.timedelta64(30, "m")
EP = PO[:, 0]
AT = A[cand]
SP0 = spr[E]
year = pd.DatetimeIndex(ts[cand]).year.to_numpy()
is_oos = ts[cand] >= SPLIT.to_datetime64()


def simulate(sl, rr, hold, mode):
    """回傳每個候選的 (出場相對 index, 出場價, 原因)；之後再套用『同時只一筆』。"""
    n = len(cand)
    sd = sl * AT
    stop = np.repeat((EP - sd)[:, None], hold, axis=1)
    if mode == "移動停損":
        runmax = np.maximum.accumulate(PH[:, :hold], axis=1)
        prev = np.concatenate([EP[:, None], runmax[:, :-1]], axis=1)          # 用前一根為止的最高價
        stop = np.maximum(stop, prev - sd[:, None])
    elif mode == "保本1R":
        reached = np.maximum.accumulate(PH[:, :hold] >= (EP + sd)[:, None], axis=1)
        prev = np.concatenate([np.zeros((n, 1), bool), reached[:, :-1]], axis=1)
        stop = np.where(prev, np.maximum(stop, EP[:, None]), stop)
    hit_sl = PL[:, :hold] <= stop
    tp = EP + rr * sd if rr > 0 else np.full(n, np.inf)
    hit_tp = PH[:, :hold] >= tp[:, None]
    first_sl = np.where(hit_sl.any(1), hit_sl.argmax(1), hold)
    first_tp = np.where(hit_tp.any(1), hit_tp.argmax(1), hold)
    xi = np.full(n, hold - 1)
    xp = PC[np.arange(n), hold - 1].copy()
    why = np.full(n, 0)                                                        # 0 時間 1 停損 2 停利
    s_m = (first_sl <= first_tp) & (first_sl < hold)                           # 同根 → 先停損
    t_m = (first_tp < first_sl) & (first_tp < hold)
    r = np.arange(n)
    j = first_sl[s_m]
    sv = stop[r[s_m], j]
    xi[s_m], why[s_m] = j, 1
    xp[s_m] = np.where(j > 0, np.minimum(PO[r[s_m], j], sv), sv)
    j = first_tp[t_m]
    xi[t_m], why[t_m] = j, 2
    xp[t_m] = np.where(j > 0, np.maximum(PO[r[t_m], j], tp[t_m]), tp[t_m])
    ok = span_ok[np.arange(n), hold - 1]
    return xi, xp, why, ok, sd


def take(xi, ok):
    """同時只持有一筆：依時間順序，前一筆出場前的訊號忽略。"""
    keep = np.zeros(len(cand), bool)
    busy = -1
    for k in range(len(cand)):
        if not ok[k] or cand[k] <= busy:
            continue
        keep[k] = True
        busy = E[k] + xi[k]
    return keep


def summarize(sel, pnl_bp, pnl_R):
    b, R = pnl_bp[sel], pnl_R[sel]
    if len(b) < 30:
        return dict(n=len(b))
    w = R[R > 0]
    lo = R[R <= 0]
    eq = np.cumsum(R)
    return dict(n=len(b), bp=b.mean(), t=b.mean() / b.std() * np.sqrt(len(b)), R=R.mean(), win=(R > 0).mean(),
                pf=w.sum() / -lo.sum() if len(lo) else np.inf, dd=(eq - np.maximum.accumulate(eq)).min())


rows = []
cache = {}
for sl, rr, hold, mode in itertools.product(SLS, RRS, HOLDS, MODES):
    if mode == "移動停損" and rr > 0:
        continue                                                               # 移動停損不再加固定停利
    xi, xp, why, ok, sd = simulate(sl, rr, hold, mode)
    keep = take(xi, ok)
    pnl = xp - EP - SP0
    bp, R = pnl / EP * 1e4, pnl / sd
    si, so, sa = summarize(keep & ~is_oos, bp, R), summarize(keep & is_oos, bp, R), summarize(keep, bp, R)
    yr = pd.Series(bp[keep]).groupby(year[keep]).mean()
    rows.append(dict(停損ATR=sl, RR=rr, 最長持有=hold, 方式=mode,
                     IS_n=si.get("n"), IS_bp=si.get("bp"), IS_t=si.get("t"), IS_R=si.get("R"),
                     OOS_n=so.get("n"), OOS_bp=so.get("bp"), OOS_t=so.get("t"), OOS_R=so.get("R"),
                     OOS_勝率=so.get("win"), OOS_PF=so.get("pf"), OOS_DD_R=so.get("dd"),
                     全期_bp=sa.get("bp"), 全期_t=sa.get("t"), 全期_R=sa.get("R"), 全期_勝率=sa.get("win"), 全期_PF=sa.get("pf"),
                     全期_DD_R=sa.get("dd"), **{f"y{y}": v for y, v in yr.items()},
                     停損比例=(why[keep] == 1).mean(), 停利比例=(why[keep] == 2).mean()))
    cache[(sl, rr, hold, mode)] = (keep, bp, R)
G = pd.DataFrame(rows)
G.round(4).to_csv("part24_grid.csv", index=False, encoding="utf-8-sig")

p("=" * 120)
p(f"第二十四部分：M1 急跌買的出場方式，共 {len(G)} 組（停損 {SLS} ATR × RR {RRS} × 最長持有 {HOLDS} 根 × 固定/保本1R/移動停損）")
p("  單位 bp（扣進場點差）；同根同時碰停損與停利 → 當作先停損；樣本內 2023/6~2024/12、樣本外 2025/1~2026/10")
p("  原本定案：停損 3 ATR、無停利、持有 48 根、固定")
p("=" * 120)
p(f"樣本內 t 與樣本外 t 的排名相關：{G.IS_t.rank().corr(G.OOS_t.rank()):+.2f}；樣本外 bp > 0 的比例 {(G.OOS_bp > 0).mean():.0%}")
base = G[(G.停損ATR == 3) & (G.RR == 0) & (G.最長持有 == 48) & (G.方式 == "固定")].iloc[0]
p(f"原本定案：IS {base.IS_bp:+.2f}bp (t {base.IS_t:+.2f}) → OOS {base.OOS_bp:+.2f}bp (t {base.OOS_t:+.2f})，全期 {base.全期_bp:+.2f}bp，勝率 {base.全期_勝率:.0%}，PF {base.全期_PF:.2f}")


def heat(df, val, title):
    t = df.pivot_table(index="停損ATR", columns="RR", values=val)
    p(f"\n--- {title}")
    p(t.round(2).to_string())


F48 = G[(G.方式 == "固定") & (G.最長持有 == 48)]
heat(F48, "IS_bp", "固定、持有 48：樣本內 bp（列 = 停損 ATR，欄 = RR；RR 0 = 不設停利）")
heat(F48, "OOS_bp", "固定、持有 48：樣本外 bp")
heat(F48, "全期_勝率", "固定、持有 48：全期勝率")
for hold in (24, 120, 240):
    heat(G[(G.方式 == "固定") & (G.最長持有 == hold)], "OOS_bp", f"固定、持有 {hold}：樣本外 bp")

p("\n--- 依單一參數平均（全部組合；樣本內 / 樣本外 bp）")
for k in ("停損ATR", "RR", "最長持有", "方式"):
    g = G.groupby(k)[["IS_bp", "OOS_bp", "全期_勝率"]].mean()
    p(g.round(3).to_string())
    p("")

p("--- 保本 / 移動停損 vs 固定（同停損、同持有、RR=0）")
cmp = G[G.RR == 0].pivot_table(index=["停損ATR", "最長持有"], columns="方式", values=["IS_bp", "OOS_bp"])
p(cmp.round(2).to_string())

cols = ["停損ATR", "RR", "最長持有", "方式", "IS_n", "IS_bp", "IS_t", "OOS_n", "OOS_bp", "OOS_t", "OOS_勝率", "OOS_PF", "全期_bp", "全期_t", "全期_DD_R",
        "y2023", "y2024", "y2025", "y2026"]
p("\n--- 樣本內 t 最高的 15 組 → 樣本外")
top = G.nlargest(15, "IS_t")
p(top[cols].round(2).to_string(index=False))
p("\n--- 四年每年都 > 0 的組合（依全期 t 排序，前 15）")
yrs = G[[c for c in G.columns if c.startswith("y20")]]
allpos = G[(yrs > 0).all(axis=1)]
p(f"共 {len(allpos)} 組")
p(allpos.nlargest(15, "全期_t")[cols].round(2).to_string(index=False))

# ---------------------------------------------------------------- 曲線
pick = [("baseline SL3 RR0 H48", (3.0, 0, 48, "固定"))]
b1 = top.iloc[0]
pick.append((f"IS-best SL{b1.停損ATR} RR{b1.RR} H{b1.最長持有} {b1.方式}", (b1.停損ATR, b1.RR, b1.最長持有, b1.方式)))
for _, r in allpos.nlargest(2, "全期_t").iterrows():
    pick.append((f"all-years+ SL{r.停損ATR} RR{r.RR} H{r.最長持有} {r.方式}", (r.停損ATR, r.RR, r.最長持有, r.方式)))
fig, ax = plt.subplots(figsize=(11, 5))
p("\n--- 曲線用的組合")
for lbl, key in pick:
    keep, bp, R = cache[key]
    ax.plot(pd.DatetimeIndex(ts[cand][keep]), np.cumsum(bp[keep]),
            label=lbl.replace("固定", "fixed").replace("保本1R", "BE1R").replace("移動停損", "trail"))
    p(f"{lbl:50s} n={keep.sum()} 全期 {bp[keep].mean():+.2f}bp R {R[keep].mean():+.3f} 勝率 {(R[keep] > 0).mean():.0%}")
ax.axvline(SPLIT, color="k", ls="--", lw=0.8)
ax.axhline(0, color="grey", lw=0.8)
ax.set_title("Part 24: M1 dip-buy exits, cumulative bp (after spread)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("part24_equity.png", dpi=110)
open("results_part24.txt", "w").write("\n".join(out) + "\n")
