"""第二十三部分：前日高 PDH（及前日低 PDL）突破單 —— 深入檢驗。

第二十二部分發現：當天第一次碰到 PDH，反彈率比假點位低約 9 個百分點（偏穿越），順勢做多 +0.23R。
這裡檢查：
  1. 參數鄰近區域（停損 0.1~0.3 ATR × 停利 1~3R / 當天收盤）
  2. 進場方式：碰到即進（停損單掛在 PDH） vs M5 收盤站上 PDH + 緩衝
  3. 是不是只是多頭漂移：同時間「隨機做多」基準、假點位、以及趨勢狀態分組（收盤 vs 20 日均線）
     → 對稱檢查：PDL 往下突破在空頭狀態下有沒有效
  4. 時段、星期、跳空、碰觸前距離等條件
  5. 逐年 / 逐月、權益曲線、回撤
資料：data/XAUUSD_M1_2023_2026.csv；broker 時間，交易日 01:00~23:59；ATR = 前 10 日日波幅平均。
成本：每筆扣進場當根點差；同一根 K 停損與停利都碰 → 算停損；碰觸那根若也碰到停損 → 算停損。
"""
import numpy as np
import pandas as pd
from load import load

OUT = "results_part23.txt"
d = load("data/XAUUSD_M1_2023_2026.csv")
d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
days = {k: g for k, g in d.groupby("day") if len(g) >= 600}
DAYS = sorted(days)
daily = pd.DataFrame({k: dict(H=g.high.max(), L=g.low.min(), C=g.close.iloc[-1], O=g.open.iloc[0]) for k, g in days.items()}).T
daily.index = pd.DatetimeIndex(daily.index)
ATR = (daily.H - daily.L).rolling(10).mean().shift(1)
SMA20 = daily.C.rolling(20).mean()
SMA50 = daily.C.rolling(50).mean()
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def sim(H, Lo, C, i, entry, d_, stop, tgt, last, check_entry_bar=True):
    """從第 i 根 K 棒（進場那根）開始模擬。tgt=None → 抱到 last。回傳價格損益（美元/盎司）。"""
    if check_entry_bar and ((Lo[i] <= stop) if d_ > 0 else (H[i] >= stop)):
        return stop - entry if d_ > 0 else entry - stop, i
    for j in range(i + 1, last + 1):
        hs = Lo[j] <= stop if d_ > 0 else H[j] >= stop
        if hs:
            return (stop - entry) * d_, j
        if tgt is not None:
            ht = H[j] >= tgt if d_ > 0 else Lo[j] <= tgt
            if ht:
                return (tgt - entry) * d_, j
    return (C[last] - entry) * d_, last


def events(level="PDH", off=0.0, entry="touch", buf=0.05, end_h=22):
    """回傳每天的事件（只取當天第一次碰觸），含後續模擬需要的陣列索引。"""
    out = []
    for a, b in zip(DAYS[:-1], DAYS[1:]):
        atr = ATR.get(b, np.nan)
        if not np.isfinite(atr):
            continue
        g = days[b]
        H, Lo, C = g.high.values, g.low.values, g.close.values
        hrs = g.index.hour.values
        base = daily.loc[a, "H"] if level == "PDH" else daily.loc[a, "L"]
        d_ = 1 if level == "PDH" else -1
        L = base + d_ * off * atr
        # 開盤已在點位外側（跳空）→ 不算「碰觸突破」，略過
        if (g.open.values[0] - L) * d_ >= 0:
            continue
        hit = (Lo <= L) & (H >= L) & (hrs < end_h)
        hit[0] = False
        if not hit.any():
            continue
        i = int(np.argmax(hit))
        if entry == "touch":
            px, ie = L, i
        else:  # 碰觸後 3 小時內，第一根 M5 收盤站上 L + buf·ATR（M1 每 5 根的最後一根）
            thr = L + d_ * buf * atr
            mins = g.index.minute.values
            ok = np.where(((mins % 5) == 4) & ((C - thr) * d_ > 0) & (np.arange(len(C)) >= i) & (np.arange(len(C)) <= i + 180) & (hrs < end_h))[0]
            if not len(ok):
                continue
            ie = int(ok[0]); px = C[ie]
        out.append(dict(day=b, a=a, i=ie, i_touch=i, px=px, L=L, d=d_, atr=atr, sp=g.spread.values[ie] * 0.01,
                        hour=int(hrs[ie]), last=len(C) - 1 if g.index[-1].hour < 23 else int(np.where(hrs >= 23)[0][0])))
    return pd.DataFrame(out)


def run(E, s, t, stop_ref="entry", hold=240, check_entry_bar=True):
    """s: 停損距離（ATR）；t: 停利倍數（R），t=None → 抱到 hold 分鐘或當天 23:00。R = 停損距離。"""
    R = []
    for e in E.itertuples():
        g = days[e.day]
        H, Lo, C = g.high.values, g.low.values, g.close.values
        ref = e.px if stop_ref == "entry" else e.L
        stop = ref - e.d * s * e.atr
        risk = (e.px - stop) * e.d
        tgt = e.px + e.d * t * risk if t is not None else None
        last = min(e.i + hold, e.last) if hold else e.last
        pnl, _ = sim(H, Lo, C, e.i, e.px, e.d, stop, tgt, last, check_entry_bar and stop_ref == "L")
        R.append((pnl - e.sp) / risk)
    E = E.copy()
    E["R"] = R
    return E


def st(r):
    r = pd.Series(r)
    if len(r) < 3:
        return "n<3"
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return f"n={len(r):4d} {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+5.2f} 勝率 {(r > 0).mean():.0%} PF {gp / max(gl, 1e-9):.2f}"


def yearly(E):
    y = pd.DatetimeIndex(E.day).year
    return " / ".join(f"{E.R[y == k].mean():+.2f}({(y == k).sum()})" for k in (2023, 2024, 2025, 2026))


p("第二十三部分：PDH / PDL 突破單深入檢驗（XAUUSD M1 2023/6~2026/10）")
p("R = 停損距離；每筆已扣點差。括號內為筆數。\n")

# ---------- 1. 參數鄰近 ----------
E = events("PDH")
p(f"PDH 事件數（開盤在 PDH 下方、當天 22 點前碰到）：{len(E)}，佔交易日 {len(E) / len(DAYS):.0%}")
p("\n=== 1. 碰到即進（停損單在 PDH），停損 = PDH 下方 s·ATR，停利 t·R，最長 4 小時 ===")
p(f"{'s\\t':8s}" + "".join(f"{('t=' + str(t)) if t else '抱4h':>12s}" for t in (1, 1.5, 2, 3, None)))
grid = {}
for s in (0.1, 0.15, 0.2, 0.3):
    row = []
    for t in (1, 1.5, 2, 3, None):
        r = run(E, s, t, stop_ref="L")
        grid[(s, t)] = r
        row.append(f"{r.R.mean():+.3f}({r.R.mean() / r.R.std() * np.sqrt(len(r)):+.1f})")
    p(f"{s:<8}" + "".join(f"{x:>12s}" for x in row))
p("  （括號為 t 值）")
p(f"  20 組中為正：{sum(v.R.mean() > 0 for v in grid.values())}/20")

BASE = (0.2, 2)
B = grid[BASE]
p(f"\n基準設定 s=0.2 ATR、t=2R：{st(B.R)}")
p(f"  逐年：{yearly(B)}")

# ---------- 2. 對照 ----------
p("\n=== 2. 是不是只是多頭漂移？ ===")
fk = pd.concat([run(events("PDH", off=o), *BASE, stop_ref="L") for o in (-0.5, -0.3, -0.15, 0.15, 0.3, 0.5)])
p(f"  假點位（PDH ± 0.15/0.3/0.5 ATR）同規則：{st(fk.R)}")
# 隨機做多：同一天、同一時間點之外的隨機時間（同時段分布），同樣停損距離與停利
rng = np.random.default_rng(0)
rl = []
for rep in range(20):
    for e in B.itertuples():
        # 從所有交易日隨機抽一天（不能只抽有碰到 PDH 的日子，否則等於偷看當天會上漲）
        dd = DAYS[int(rng.integers(10, len(DAYS)))]
        atr = ATR.get(dd, np.nan)
        g = days[dd]
        H, Lo, C = g.high.values, g.low.values, g.close.values
        hrs = g.index.hour.values
        cand = np.where(hrs == e.hour)[0]
        if not len(cand) or not np.isfinite(atr):
            continue
        j = int(rng.choice(cand))
        last = len(C) - 1 if g.index[-1].hour < 23 else int(np.where(hrs >= 23)[0][0])
        px = C[j]
        risk = 0.2 * atr
        pnl, _ = sim(H, Lo, C, j, px, 1, px - risk, px + 2 * risk, min(j + 240, last), False)
        rl.append((pnl - g.spread.values[j] * 0.01) / risk)
p(f"  隨機做多（隨機交易日、與訊號相同的小時、同停損停利，20 次）：{st(rl)}")

# 趨勢狀態
tr = pd.DataFrame({"up20": daily.C > SMA20, "up50": daily.C > SMA50}).shift(0)
B = B.join(tr, on="a")
for col in ("up20", "up50"):
    up, dn = B[B[col] == True], B[B[col] == False]
    p(f"  前一日收盤 vs {col[2:]} 日均線：上方 {st(up.R)} | 下方 {st(dn.R)}")

# PDL 對稱
EL = events("PDL")
BL = run(EL, *BASE, stop_ref="L").join(tr, on="a")
p(f"\n  PDL 往下突破做空：{st(BL.R)}  逐年 {yearly(BL)}")
for col in ("up20", "up50"):
    p(f"    {col[2:]} 日均線下方（空頭狀態）：{st(BL[BL[col] == False].R)} | 上方：{st(BL[BL[col] == True].R)}")

# ---------- 3. 進場方式 ----------
p("\n=== 3. 進場方式（s=0.2 ATR，t=2R）===")
for buf in (0.0, 0.05, 0.1):
    Em = events("PDH", entry="m5", buf=buf)
    for ref in ("L", "entry"):
        r = run(Em, 0.2, 2, stop_ref=ref)
        p(f"  M5 收盤站上 PDH+{buf} ATR，停損距{'PDH' if ref == 'L' else '進場價'} 0.2 ATR：{st(r.R)}  逐年 {yearly(r)}")

# ---------- 4. 條件 ----------
p("\n=== 4. 條件分組（基準設定）===")
B["dow"] = pd.DatetimeIndex(B.day).dayofweek
B["gapd"] = [(e.L - days[e.day].open.values[0]) / e.atr for e in B.itertuples()]  # 開盤離 PDH 多遠
B["prevup"] = [daily.loc[a, "C"] > daily.loc[a, "O"] for a in B.a]
B["pclose_pos"] = [(daily.loc[a, "C"] - daily.loc[a, "L"]) / (daily.loc[a, "H"] - daily.loc[a, "L"]) for a in B.a]
for lab, mask in [("01-08 點（亞洲）", B.hour < 8), ("08-15 點（倫敦）", (B.hour >= 8) & (B.hour < 15)), ("15-22 點（紐約）", B.hour >= 15)]:
    p(f"  {lab:16s} {st(B[mask].R)}")
for k, nm in enumerate("一二三四五"):
    p(f"  星期{nm}           {st(B[B.dow == k].R)}")
q = pd.qcut(B.gapd, 3, labels=["開盤離 PDH 近", "中", "遠"])
for lab, g in B.groupby(q, observed=True):
    p(f"  {lab:16s} {st(g.R)}")
p(f"  前一天收紅（C>O）  {st(B[B.prevup].R)} | 收黑 {st(B[~B.prevup].R)}")
q = pd.qcut(B.pclose_pos, 3, labels=["前日收在區間下段", "中段", "上段"])
for lab, g in B.groupby(q, observed=True):
    p(f"  {lab:16s} {st(g.R)}")

# ---------- 5. 權益 ----------
p("\n=== 5. 基準設定的權益、回撤、逐月 ===")
B = B.sort_values("day")
eq = B.R.cumsum()
p(f"  總計 {B.R.sum():+.1f}R，最大回撤 {(eq.cummax() - eq).max():.1f}R，最長連虧 "
  f"{max((len(list(g)) for k, g in __import__('itertools').groupby(B.R < 0) if k), default=0)} 筆")
m = B.groupby(pd.DatetimeIndex(B.day).to_period("M")).R.sum()
p(f"  正報酬月份 {(m > 0).sum()}/{len(m)}；季度：")
qq = B.groupby(pd.DatetimeIndex(B.day).to_period("Q")).R.agg(["sum", "count"])
p("  " + "  ".join(f"{k}:{v['sum']:+.1f}({int(v['count'])})" for k, v in qq.iterrows()))
B[["day", "hour", "px", "L", "atr", "sp", "R"]].to_csv("part23_pdh_trades.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(pd.DatetimeIndex(B.day), eq.values, label="PDH breakout (s=0.2ATR, 2R)")
    fk = fk.sort_values("day")
    ax[0].plot(pd.DatetimeIndex(fk.day), fk.R.cumsum().values / 6, label="placebo levels (avg of 6)", alpha=.7)
    ax[0].set_ylabel("cum R"); ax[0].legend(); ax[0].grid(alpha=.3)
    px = daily.C
    ax[1].plot(px.index, px.values, color="goldenrod"); ax[1].set_ylabel("XAUUSD"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("part23_pdh_equity.png", dpi=110)
except ImportError:
    pass

open(OUT, "w").write("\n".join(lines) + "\n")
