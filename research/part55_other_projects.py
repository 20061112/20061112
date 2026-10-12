"""第五十五部分：把使用者其他分支裡的東西套進目前的組合（IB 核心 + M15 最終版 + ATR 熵）。

來源：
  claude/zen-pasteur-xz0bdu  FINAL_STRATEGY_OW.md（OW 版 IB 突破）的三個改良，套在 IB 核心上（OW 本身多一個「前日價值區外」篩選，這裡不加）：
    W  IB 寬度兩端：IB 寬 ≤ 0.4×ATR10，且 ≥ 0.6×前 20 個交易日同一起點 IB 寬的中位數（先篩選再去重）
    BE 2R 保本：浮盈 ≥ 2R 後從下一根起停損移到進場價
    SZ 依突破快慢分倉：IB 結束到進場 ≤ 0.5 小時 3 份、0.5~1 小時 2 份、1~3 小時 1 份
  ccr-754cdf47-jr7hab  STEEP_SPEC.md（很陡順勢，M15）：
    cosθ = 1/√(1+tan²θ)，tanθ = |C[t]−C[t−48]| / (ATR14[t]·√48)；z = (cosθ − EWMA1000) / EWMstd1000；
    z ≤ −2 且沒有持倉 → 下一根 M15 開盤順 48 根方向進場；停損 1.5×ATR14；第 48 根 M15 收盤出場；不設停利
    這裡用 M1 執行（進場 = 下一根 M15 第一分鐘的開盤價，停損用 M1 高低判斷）
其他分支看過但不採用（理由見 README）：ATR 時間尺度 z、回復力切換、小 K 中點、K 線角度共振、Fib / 諧波、SMC / POC / FVG、網格。
成本一律：點差 + 每筆 0.3 美元/盎司滑價。數量單位：1 份 = 0.01 口 = 1 盎司。
"""
import inspect
import numpy as np
import pandas as pd
import part54_core_ib_combo as P54
from part54_core_ib_combo import CI, IDX, SLIP, minute_eq

lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def pf(v):
    v = np.asarray(v)
    return v[v > 0].sum() / -v[v < 0].sum()


# ---------------- IB 核心：不去重的事件（OW 的寬度篩選要先篩選再去重）----------------
src = inspect.getsource(CI.events).replace('return E.sort_values(["t_in", "s0"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)',
                                           'return E.sort_values(["t_in", "s0"]).reset_index(drop=True)')
ns = dict(CI.__dict__)
exec(src.replace("def events(", "def events_all("), ns)
E_all = ns["events_all"]()
dedup = lambda E: E.drop_duplicates(["t_in", "side"]).reset_index(drop=True)

# 所有 IB（不論有沒有突破）的寬度，做「前 20 天同一起點的中位數」
rows = []
for day in CI.DAYS:
    t, mins, O, H, L, C_, S = CI.ARR[day]
    for s0 in range(120, 631, 30):
        m = (mins >= s0) & (mins < s0 + 60)
        if m.sum() >= 48:
            rows.append((day, s0, H[m].max() - L[m].min()))
IBW = pd.DataFrame(rows, columns=["day", "s0", "w"])
IBW["med20"] = IBW.groupby("s0").w.transform(lambda s: s.rolling(20, min_periods=10).median().shift(1))
E_all = E_all.merge(IBW[["day", "s0", "med20"]], on=["day", "s0"], how="left")
E_all["width"] = E_all.ibh - E_all.ibl
wmask = (E_all.width <= 0.4 * E_all.atr) & (E_all.width >= 0.6 * E_all.med20)


def prep(X):
    X = X[(X.peff <= CI.PEFF_MAX) & (X.day >= "2023-06-15")].copy()
    gi, gj = [], []
    for r in X.itertuples():
        t = CI.ARR[r.day][0]
        gi.append(IDX.get_indexer([t[r.i0]])[0]); gj.append(IDX.get_indexer([t[int(r.j)]])[0])
    X["gi"], X["gj"] = gi, gj
    X = X[(X.gi >= 0) & (X.gj >= 0)].copy()
    X["net"] = X.pnl - SLIP
    delay = X.min_in - (X.s0 + 60)
    X["q"] = np.where(delay < 30, 3, np.where(delay < 60, 2, 1))
    return X


VAR = {}
VAR["IB 核心"] = prep(CI.simulate(dedup(E_all)))
VAR["IB + W 寬度"] = prep(CI.simulate(dedup(E_all[wmask])))
VAR["IB + BE 2R保本"] = prep(CI.simulate(dedup(E_all), be=2))
VAR["IB + W + BE"] = prep(CI.simulate(dedup(E_all[wmask]), be=2))
idx = P54.idx


def daily(X, q):
    return (X.net * q).groupby(X.day).sum().reindex(idx, fill_value=0)


def parts_q(X, qv):
    """依每筆口數分組（minute_eq 每組一個口數）。"""
    out = []
    for qq in np.unique(qv):
        m = qv == qq
        out.append((X.gi.values[m], X.gj.values[m], X.side.values[m], X.lvl.values[m], X.net.values[m], qq))
    return out


p("第五十五部分：使用者其他分支的改良 × 目前組合（點差 + 0.3 滑價；1 份 = 0.01 口）")
p("\n=== (1) OW 版的三個改良套在 IB 核心上（每筆 1 份）===")
p("  逐期每筆 / PF（1 盎司）：23/6~24/10 是 IB 核心與 OW 改良都沒用來選參數的期間（OW 的寬度篩選是在 M15 2023~2026 看過後才定）")
for k, X in VAR.items():
    per = []
    for a, b in [("2023-06-01", "2024-11-01"), ("2024-11-01", "2026-01-01"), ("2026-01-01", "2027-01-01")]:
        g = X[(X.day >= a) & (X.day < b)]
        per.append(f"{g.net.mean():+.2f}/{pf(g.net):.2f}")
    p(f"  {k:14s} {len(X):5d} 筆 每天 {len(X) / X.day.nunique():.1f} 每筆 R {X.R.mean():+.3f} | 23/6~24/10 {per[0]}  24/11~25/12 {per[1]}  2026 {per[2]}")
p("  突破快慢（IB 結束到進場）每筆 / PF：" + "；".join(
    f"{lab} {g.net.mean():+.2f}/{pf(g.net):.2f}（{len(g)} 筆；23/6~24/10 {g[g.day < '2024-11-01'].net.mean():+.2f}）"
    for lab, g in VAR["IB 核心"].groupby(VAR["IB 核心"].q.map({3: "≤0.5h", 2: "0.5~1h", 1: "1~3h"}))))

p("\n  單獨（總曝險約 0.01 口 / 筆；SZ 版本把 3/2/1 份除以平均份數，讓平均口數與 1 份相同）")
P54.lines = lines   # 讓 show() 寫進本檔結果
show = P54.show
for k, X in VAR.items():
    show(k + " 0.01", [(X.gi.values, X.gj.values, X.side.values, X.lvl.values, X.net.values, 1)], daily(X, 1))
    qn = X.q.values / X.q.mean()
    show(k + " + SZ", parts_q(X, np.round(qn, 4)), (X.net * qn).groupby(X.day).sum().reindex(idx, fill_value=0))

# ---------------- (2) 很陡順勢 ----------------
m15 = pd.read_csv("data/XAUUSD_M15_2023_2026.csv", sep="\t")
m15.index = pd.to_datetime(m15["<DATE>"] + " " + m15["<TIME>"], format="%Y.%m.%d %H:%M:%S")
O15, H15, L15, C15 = (m15[c].values for c in ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"])
pc = np.r_[np.nan, C15[:-1]]
tr = np.nanmax(np.c_[H15 - L15, np.abs(H15 - pc), np.abs(L15 - pc)], axis=1)
atr = pd.Series(tr).rolling(14).mean().values
net48 = C15 - np.r_[np.full(48, np.nan), C15[:-48]]
tan = np.abs(net48) / (atr * np.sqrt(48))
cos = pd.Series(1 / np.sqrt(1 + tan ** 2))
z = ((cos - cos.ewm(span=1000, adjust=False).mean()) / cos.ewm(span=1000, adjust=False).std()).values
z[:1100] = np.nan
O1, H1, L1, C1 = (P54.P.W.B.d[c].values for c in ["open", "high", "low", "close"])
SP1 = P54.P.SP
t15 = m15.index
rows, busy_until = [], pd.Timestamp("1900-01-01")
for t in np.flatnonzero(z <= -2):
    if t + 49 >= len(t15) or not np.isfinite(atr[t]) or net48[t] == 0:
        continue
    t_in, t_out = t15[t + 1], t15[t + 49]          # 進場 = 下一根開盤；出場 = 第 48 根收盤（= 第 49 根開盤前一分鐘）
    if t_in < busy_until or t_in < IDX[0] or t_out > IDX[-1]:
        continue
    gi = IDX.searchsorted(t_in)
    gj_end = IDX.searchsorted(t_out) - 1
    if gi >= len(IDX) or gj_end <= gi:
        continue
    s = int(np.sign(net48[t])); L = O1[gi]; risk = 1.5 * atr[t]; st = L - s * risk
    lo, hi = L1[gi:gj_end + 1], H1[gi:gj_end + 1]
    hit = np.flatnonzero(lo <= st) if s == 1 else np.flatnonzero(hi >= st)
    if len(hit):
        j = gi + hit[0]; px = st if hit[0] == 0 else (min(O1[j], st) if s == 1 else max(O1[j], st)); why = "停損"
    else:
        j = gj_end; px = C1[j]; why = "時間"
    busy_until = IDX[j]
    net = s * (px - L) - SP1[gi] - SLIP
    rows.append((IDX[gi], IDX[j], gi, j, s, L, net, net / risk, risk, why))
S = pd.DataFrame(rows, columns=["進場時間", "出場時間", "gi", "gj", "side", "lvl", "net", "R", "risk", "why"])
S["day"] = S.進場時間.dt.normalize()
p(f"\n=== (2) 很陡順勢（STEEP_SPEC，M15 訊號、M1 執行；{S.進場時間.min().date()} ~ {S.進場時間.max().date()}）===")
p(f"  {len(S)} 筆，每週 {len(S) / ((S.進場時間.max() - S.進場時間.min()).days / 7):.1f} 筆，勝率 {(S.R > 0).mean():.0%}，停損出場 {(S.why == '停損').mean():.0%}，"
  f"每筆 R {S.R.mean():+.3f}（不扣滑價 {((S.net + SLIP) / S.risk).mean():+.3f}）")
p("  逐年 每筆 R / PF（美元，扣滑價）：" + "；".join(f"{y} {g.R.mean():+.2f} / {pf(g.net):.2f}（{len(g)} 筆）" for y, g in S.groupby(S.進場時間.dt.year)))
# 固定美元風險：每筆盎司 = k / 停損距離，k 讓平均盎司 = 1
k_r = 1 / np.mean(1 / S.risk.values)
qr = k_r / S.risk.values
st_d1 = daily(S.assign(gi=S.gi, gj=S.gj), 1)
st_dr = (S.net * qr).groupby(S.day).sum().reindex(idx, fill_value=0)
ib_best = VAR["IB + W + BE"]
ib_d = daily(ib_best, 1)
me_d = P54.me_d / 2              # P54.me_d 是 0.02 口（2 盎司）
d_all = pd.DataFrame({"IB": ib_d, "M15熵": me_d, "陡固定": st_d1, "陡風險": st_dr})
p("  日損益相關（全部日子）：\n" + d_all.corr().round(2).to_string())
show("很陡 固定 0.01", [(S.gi.values, S.gj.values, S.side.values, S.lvl.values, S.net.values, 1)], st_d1)
show("很陡 固定美元風險（平均 0.01）", parts_q(S.assign(gi=S.gi), np.round(qr, 3)), st_dr)

# ---------------- (3) 組合 ----------------
p("\n=== (3) 組合（份數 = 0.01 口的倍數）===")
ME = P54.ME
me_part = lambda q: (ME.i.values, ME.j.values, ME.dir.values, ME.entry.values, ME.net.values, q)
ib0 = VAR["IB 核心"]
ib_part = lambda X, q: (X.gi.values, X.gj.values, X.side.values, X.lvl.values, X.net.values, q)
st_part = lambda q: (S.gi.values, S.gj.values, S.side.values, S.lvl.values, S.net.values, q)
show("目前：IB 1 + M15熵 2", [ib_part(ib0, 1), me_part(2)], daily(ib0, 1) + 2 * me_d)
show("IB(W+BE) 1 + M15熵 2", [ib_part(ib_best, 1), me_part(2)], ib_d + 2 * me_d)
show("IB(W+BE) 2 + M15熵 2", [ib_part(ib_best, 2), me_part(2)], 2 * ib_d + 2 * me_d)
show("IB 1 + M15熵 2 + 很陡 1", [ib_part(ib0, 1), me_part(2), st_part(1)], daily(ib0, 1) + 2 * me_d + st_d1)
show("IB(W+BE) 1 + M15熵 2 + 很陡 1", [ib_part(ib_best, 1), me_part(2), st_part(1)], ib_d + 2 * me_d + st_d1)
show("IB(W+BE) 2 + M15熵 2 + 很陡 1", [ib_part(ib_best, 2), me_part(2), st_part(1)], 2 * ib_d + 2 * me_d + st_d1)
show("IB(W+BE) 2 + M15熵 2 + 很陡 2", [ib_part(ib_best, 2), me_part(2), st_part(2)], 2 * ib_d + 2 * me_d + 2 * st_d1)
open("results_part55.txt", "w").write("\n".join(lines) + "\n")
