"""第三十三部分：連虧上限 → 轉影子模式。

基礎訊號：final_m15_signals.csv（A + B + 50MA 斜率，上限 8 的全部影子訊號，含權益濾網狀態）。
規則：依出場時間順序，實際下單的單連續虧損達 K 筆 → 停手（新訊號只記錄不下單；已持有的單照常出場）。
恢復條件：
  win1    停手後第一筆影子單獲利 → 恢復
  win2    停手後影子單連續 2 筆獲利 → 恢復
  plus2R  停手後影子單累積 ≥ +2R → 恢復
  nextday 隔天恢復
基礎模式：有 / 沒有第三十一部分的權益濾網（50 筆均線）。
指標：每筆固定 1R；總 R、MDD（日結）、總/MDD、最長連虧（實際下單、依出場順序）；訓練 2023~2024 / 驗證 2025~2026。
"""
import numpy as np
import pandas as pd

S = pd.read_csv("final_m15_signals.csv", parse_dates=["進場時間", "出場時間"])
S["train"] = S.進場時間 < "2025-01-01"
lines = []


def p(s=""):
    print(s)
    lines.append(s)


def run(base, K, rule):
    n = len(S)
    ent = [(t, 1, k) for k, t in enumerate(S.進場時間)]
    ext = [(t, 0, k) for k, t in enumerate(S.出場時間)]
    evs = sorted(ent + ext)  # 同一時間：先出場(0) 再進場(1)
    R = S.R.values
    day = S.進場時間.dt.normalize().values
    take = np.zeros(n, bool)
    paused, streak, acc, wins, pday, npause = False, 0, 0.0, 0, None, 0
    for t, typ, k in evs:
        if typ == 1:
            if paused and rule == "nextday" and day[k] > pday:
                paused = False
            if base[k] and not paused:
                take[k] = True
            continue
        if take[k]:
            streak = streak + 1 if R[k] < 0 else 0
            if K and streak >= K and not paused:
                paused, streak, acc, wins, pday = True, 0, 0.0, 0, np.datetime64(pd.Timestamp(t).normalize())
                npause += 1
        elif paused and base[k]:
            acc += R[k]
            wins = wins + 1 if R[k] > 0 else 0
            if (rule == "win1" and wins >= 1) or (rule == "win2" and wins >= 2) or (rule == "plus2R" and acc >= 2):
                paused = False
    return take, npause


def max_streak(x):
    r = x.sort_values("出場時間").R.values
    best = cur = 0
    for v in r:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best


def stat(x):
    d = x.groupby(x.進場時間.dt.date).R.sum()
    e = d.cumsum()
    m = (e.cummax() - e).max()
    return x.R.sum(), m


def row(take, lab, npause):
    x = S[take]
    out = []
    for part in (x[x.train], x[~x.train], x):
        tot, m = stat(part)
        out.append(f"{tot:+6.0f} {m:4.0f} {tot / m:5.1f}")
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    tot, m = stat(x)
    p(f"{lab:26s} {len(x):5d} {x.R.mean():+.3f} {max_streak(x):4d} {npause:5d} | {out[0]} | {out[1]} | {out[2]}  年化/MDD {tot / yrs / m:4.1f}")


HDR = f"{'設定':26s} {'筆數':>5s} {'每筆R':>6s} {'連虧':>4s} {'停手次':>5s} | {'訓練R':>6s} {'MDD':>4s} {'比值':>5s} | {'驗證R':>6s} {'MDD':>4s} {'比值':>5s} | {'全期R':>6s} {'MDD':>4s} {'比值':>5s}"
p("第三十三部分：連虧上限 → 轉影子模式（M15 最終版訊號，每筆 1R）")
p("連虧 = 實際下單依出場順序的最長連續虧損筆數；停手次 = 觸發幾次\n")
for bname, base in (("沒有權益濾網", np.ones(len(S), bool)), ("有權益濾網", S.實際下單.values)):
    p(f"=== 基礎：{bname} ===")
    p(HDR)
    take, npz = run(base, 0, None)
    row(take, "不設連虧上限", npz)
    for rule in ("win1", "win2", "plus2R", "nextday"):
        for K in (5, 8, 10, 15, 20):
            take, npz = run(base, K, rule)
            row(take, f"連虧 {K} 停手，{rule}", npz)
    p()

# 連虧的結構：同一天一起停損的比例
x = S[S.實際下單].sort_values("出場時間").reset_index(drop=True)
neg = x.R < 0
grp = (neg != neg.shift()).cumsum()
runs = x[neg].groupby(grp[neg])
info = runs.agg(n=("R", "size"), days=("進場時間", lambda s: s.dt.date.nunique()), hrs=("出場時間", lambda s: (s.max() - s.min()).total_seconds() / 3600))
big = info[info.n >= 10]
p("=== 連虧的結構（有權益濾網的實際下單）===")
p(f"  連虧 ≥ 10 筆的段落 {len(big)} 段；每段跨越天數中位數 {big.days.median():.0f} 天、時間跨度中位數 {big.hrs.median():.0f} 小時；最長一段 {info.n.max()} 筆、跨 {info.loc[info.n.idxmax(), 'days']} 天")
pl = (x.R < 0).mean()
p(f"  若每筆獨立（虧損機率 {pl:.0%}），{len(x)} 筆中最長連虧的期望約 {np.log(len(x)) / np.log(1 / pl):.0f} 筆 → 實際更長是因為同時持有的單一起停損")
open("results_part33.txt", "w").write("\n".join(lines) + "\n")

# 為什麼停手沒有幫助：連虧 K 筆之後，下一個「新進場」的訊號平均賺多少？
x = S[S.實際下單].sort_values("出場時間").reset_index(drop=True)
lines2 = ["", "=== 連虧 K 筆後（依出場時間），之後 20 個新進場訊號的平均 R（有權益濾網的實際下單）==="]
r = x.R.values
streaks = np.zeros(len(x), int)
c = 0
for k, v in enumerate(r):
    c = c + 1 if v < 0 else 0
    streaks[k] = c
ent = S[S.實際下單].sort_values("進場時間").reset_index(drop=True)
et = ent.進場時間.values
for K in (5, 10, 15, 20):
    idx = np.where(streaks == K)[0]
    nxt = []
    for k in idx:
        t0 = np.datetime64(x.出場時間.iloc[k])
        j = np.searchsorted(et, t0, side="right")
        nxt.extend(ent.R.values[j:j + 20])
    lines2.append(f"  連虧 {K:2d} 筆後（{len(idx)} 次）：之後 20 筆平均 {np.mean(nxt):+.3f}R（全部平均 {ent.R.mean():+.3f}R）")
print("\n".join(lines2))
open("results_part33.txt", "a").write("\n".join(lines2) + "\n")
